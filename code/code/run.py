#!/usr/bin/env python3
import time
begin_time=time.time()

"""Unified full-visual 2B plus mandatory trained-27B selective review entry."""
import argparse,json,os,sys,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
if importlib.util.find_spec('qwen_vl_utils') is None:
    wheels=sorted((ROOT/'vendor/wheels').glob('qwen_vl_utils-*.whl'))
    if wheels:sys.path.insert(0,str(wheels[-1]))
for k,v in {'TRACK4_FINAL_OFFLINE':'1','TRACK4_DISABLE_SECRETS':'1',
            'TRACK4_RETRIEVAL_BACKEND':'local','TRACK4_LOCAL_QWEN_VL_DEVICE':'cuda',
            'TRACK4_OFFICIAL_EMBEDDING_MODEL_ID':'Qwen3-VL-Embedding-2B',
            'TRACK4_LOCAL_IMAGE_BATCH':'4','TRACK4_EMBED_PREP_WORKERS':'8',
            'TRACK4_LOCAL_QWEN_VL_EMBEDDING_PATH':str(ROOT/'model_assets/bases/embedding_2b'),
            'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','PYTHONUTF8':'1',
            'PYTHONUNBUFFERED':'1'}.items():os.environ.setdefault(k,v)
os.environ['CUDA_VISIBLE_DEVICES']='0'


def metadata_for(input_dir,output_dir,explicit):
    if explicit:return Path(explicit).resolve()
    path=input_dir/'metadata.json'
    if path.is_file():return path
    # Asset identity is supplied as context; images never decide bridge identity
    # by a disease route or a remembered test answer.
    context_path=input_dir/'asset_context.json'
    context=json.loads(context_path.read_text(encoding='utf-8')) if context_path.is_file() else {}
    bridge_name=str(context.get('bridgeName') or os.environ.get('TRACK4_BRIDGE_NAME') or '')
    rows=[]
    for cat in ('轨道','桥梁'):
        directory=input_dir/cat
        if not directory.is_dir():continue
        for image in sorted(directory.rglob('*')):
            if not image.is_file() or image.suffix.lower() not in {'.jpg','.jpeg','.png','.webp','.bmp'}:continue
            if cat=='桥梁' and not bridge_name:
                raise ValueError('Bridge identity is missing: supply metadata.json, asset_context.json or TRACK4_BRIDGE_NAME')
            rows.append({'questionCategory':cat,'bridgeName':bridge_name if cat=='桥梁' else '',
                         'filename':image.name,'imageRelativePath':str(image.relative_to(input_dir))})
    if not rows:raise FileNotFoundError('No image inputs or metadata found')
    p=output_dir/'runtime_metadata.json';p.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    return p


def main():
    p=argparse.ArgumentParser(description='Full-image inspection with trained teacher review')
    p.add_argument('--input-dir',default=os.environ.get('TRACK4_INPUT_DIR',''))
    p.add_argument('--metadata',default=os.environ.get('TRACK4_METADATA',''))
    p.add_argument('--output-dir',default=os.environ.get('TRACK4_OUTPUT_DIR',''))
    p.add_argument('--config',default=os.environ.get('TRACK4_CONFIG',str(ROOT/'configs/submission_unified.yaml')))
    p.add_argument('--dry-run',action='store_true')
    # Backwards-compatible CLI spelling; no resident service is needed.
    p.add_argument('--skip-server',action='store_true')
    args=p.parse_args()
    import yaml
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    mode=str(cfg.get('run_mode') or '').strip().lower()
    if not mode:
        mode='risk_gate_teacher' if cfg.get('selective_review',{}).get('enabled') else '2b_only'
    if mode not in {'2b_only','risk_gate','risk_gate_teacher'}:
        raise ValueError('Unsupported run_mode: '+mode)
    if not cfg.get('full_visual',{}).get('enabled'):
        raise ValueError('full_visual.enabled must be true')
    review_enabled=bool(cfg.get('selective_review',{}).get('enabled'))
    gate_enabled=bool(cfg.get('risk_gate',{}).get('enabled'))
    if mode in {'2b_only','risk_gate'} and review_enabled:
        raise ValueError('Teacher review is not allowed in '+mode+' mode')
    if mode=='risk_gate' and not gate_enabled:
        raise ValueError('risk_gate mode requires risk_gate.enabled=true')
    if mode=='risk_gate_teacher':
        if not review_enabled:
            raise ValueError('risk_gate_teacher mode requires selective_review.enabled=true')
        if cfg['selective_review'].get('max_records',0)<=0 or cfg['selective_review'].get('max_fraction',0)<=0:
            raise ValueError('risk_gate_teacher mode requires a positive teacher review budget')
        if not gate_enabled:
            raise ValueError('risk_gate_teacher mode requires risk_gate.enabled=true')
    if cfg.get('full_visual',{}).get('aerial_healthy_gate'):
        raise ValueError('Aerial forced-healthy gate is prohibited in this candidate')
    if cfg.get('heads_only',{}).get('embedder')!='qwen3_vl_embedding_2b':
        raise ValueError('Only the frozen 2B image backend is supported')
    cfg['run_mode']=mode
    input_dir=Path(args.input_dir).resolve() if args.input_dir else Path('/dataset/决赛数据集1/赛题4/线上测试集')
    output_dir=Path(args.output_dir).resolve() if args.output_dir else ROOT/'output'
    if args.dry_run:
        print(json.dumps({'run_mode':mode,'full_visual':True,'risk_gate':gate_enabled,
                          'selective_review':review_enabled,'wiki_tools':False,
                          'input_dir':str(input_dir),'output_dir':str(output_dir),'config':str(args.config)}));return
    output_dir.mkdir(parents=True,exist_ok=True)
    if (output_dir/'result.json').exists():raise FileExistsError('Use a fresh output directory')
    metadata=metadata_for(input_dir,output_dir,args.metadata)
    from track4_contract_agent.runtime.full_visual import run_full_visual
    info=run_full_visual(input_dir,metadata,output_dir,cfg,ROOT)
    import importlib.metadata as md
    teacher=None
    if mode=='risk_gate_teacher':
        teacher=Path(cfg['selective_review']['base_model'])
        if not teacher.is_absolute():teacher=ROOT/teacher
    audit={'run_mode':mode,'risk_gate_enabled':gate_enabled,
           'selective_review_enabled':review_enabled,
           'python_prefix':sys.prefix,'base_prefix':sys.base_prefix,'code_root':str(ROOT),
           'embedding_root':str(Path(os.environ['TRACK4_LOCAL_QWEN_VL_EMBEDDING_PATH']).resolve()),
           'teacher_root':str(teacher.resolve()) if teacher is not None else None,'python_path':sys.path,
           'module_files':{name:str(getattr(sys.modules.get(name),'__file__','')) for name in
                           ['torch','transformers','numpy','tongagents','PIL','yaml']},
           'versions':{name:md.version(name) for name in ['torch','transformers','peft','accelerate','safetensors','tongagents']}}
    (output_dir/'deployment_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    elapsed=time.time()-begin_time
    (output_dir/'infer_time.json').write_text(json.dumps({'infer_time':elapsed,'inter_time':elapsed},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'run_mode':mode,'n_rows':info['n_rows'],
                      'risk_gate':info.get('risk_gate'),
                      'review':info.get('selective_review'),
                      'infer_time':elapsed,'output_dir':str(output_dir)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
