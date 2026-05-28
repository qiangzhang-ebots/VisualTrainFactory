from ultralytics import YOLO
import time

name = 'train_10'


def exportYoloOnnx():
    model = YOLO('./runs/pose/'+name+'/weights/best.pt')  # load a pretrained model (recommended for training)
    model.export(format="onnx")

def exportHRNetOnnx():
    import copy
    import json
    from pathlib import Path

    last_checkpoint_path = Path(f'./runs/HRNet/{name}/last_checkpoint').expanduser().resolve()
    if not last_checkpoint_path.exists():
        raise FileNotFoundError(f'last_checkpoint file not found: {last_checkpoint_path}')

    checkpoint_text = last_checkpoint_path.read_text(encoding='utf-8').strip()
    checkpoint_path = Path(checkpoint_text).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (last_checkpoint_path.parent / checkpoint_path).resolve()
    else:
        checkpoint_path = checkpoint_path.resolve()

    checkpoint = str(checkpoint_path)
    save = str(checkpoint_path.with_suffix('.onnx'))
    model_config = None
    img = None
    device = 'cuda:0'
    verbose = True

    import mmengine
    from mmdeploy.apis import torch2onnx

    project_root = Path(__file__).resolve().parent

    def resolve_path(raw_path):
        return Path(raw_path).expanduser().resolve()

    def resolve_config_relative_path(raw_path, model_cfg_path):
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()

        for root in (Path.cwd(), project_root, model_cfg_path.parent):
            resolved = (root / candidate).resolve()
            if resolved.exists():
                return resolved
        return (project_root / candidate).resolve()

    def infer_model_config(checkpoint_path):
        candidates = sorted(
            checkpoint_path.parent.glob('*.py'),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f'No model config .py file found in checkpoint directory: {checkpoint_path.parent}'
            )
        return candidates[0]

    def resolve_checkpoint_path(checkpoint_path):
        if checkpoint_path.exists():
            return checkpoint_path

        last_checkpoint = checkpoint_path.parent / 'last_checkpoint'
        if last_checkpoint.exists():
            redirected = Path(last_checkpoint.read_text(encoding='utf-8').strip()).expanduser().resolve()
            if redirected.exists():
                if verbose:
                    print(f'Warning: checkpoint not found, fallback to last_checkpoint: {redirected}')
                return redirected

        raise FileNotFoundError(f'Checkpoint file not found: {checkpoint_path}')

    def extract_input_size(model_cfg):
        codec = model_cfg.codec
        if isinstance(codec, (list, tuple)):
            codec = codec[-1]
        input_size = codec.get('input_size')
        if input_size is None or len(input_size) != 2:
            raise ValueError('Failed to read codec.input_size from the model config.')
        return int(input_size[0]), int(input_size[1])

    def build_test_pipeline(model_cfg):
        input_w, input_h = extract_input_size(model_cfg)
        codec = model_cfg.codec
        if isinstance(codec, (list, tuple)):
            codec = codec[-1]
        use_udp = codec.get('type') == 'UDPHeatmap'
        return [
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(input_w, input_h), use_udp=use_udp),
            dict(type='PackPoseInputs'),
        ]

    def prepare_model_cfg(model_cfg_path):
        model_cfg = mmengine.Config.fromfile(str(model_cfg_path))
        export_cfg = copy.deepcopy(model_cfg)

        if export_cfg.get('test_dataloader') is None:
            train_dataloader = export_cfg.get('train_dataloader')
            if train_dataloader is None or train_dataloader.get('dataset') is None:
                raise ValueError(
                    'Model config does not contain train_dataloader.dataset for export fallback.'
                )
            dataset_cfg = copy.deepcopy(train_dataloader.dataset)
            dataset_cfg.pipeline = build_test_pipeline(export_cfg)
            export_cfg.test_dataloader = dict(
                batch_size=1,
                num_workers=0,
                persistent_workers=False,
                drop_last=False,
                sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
                dataset=dataset_cfg,
            )
        elif export_cfg.test_dataloader.get('dataset') is None:
            raise ValueError('test_dataloader exists but has no dataset config.')
        else:
            export_cfg.test_dataloader = copy.deepcopy(export_cfg.test_dataloader)
            export_cfg.test_dataloader.dataset = copy.deepcopy(export_cfg.test_dataloader.dataset)
            export_cfg.test_dataloader.dataset.pipeline = build_test_pipeline(export_cfg)

        if export_cfg.get('val_dataloader') is None:
            export_cfg.val_dataloader = copy.deepcopy(export_cfg.test_dataloader)
        if export_cfg.get('test_evaluator') is None:
            export_cfg.test_evaluator = None
        if export_cfg.get('val_evaluator') is None:
            export_cfg.val_evaluator = None

        if getattr(export_cfg.model, 'test_cfg', None) is None:
            export_cfg.model.test_cfg = dict()
        export_cfg.model.test_cfg['flip_test'] = False
        return export_cfg

    def infer_image_from_dataset(model_cfg, model_cfg_path):
        dataset_cfg = None
        for dataloader_name in ('test_dataloader', 'val_dataloader', 'train_dataloader'):
            dataloader_cfg = model_cfg.get(dataloader_name)
            if dataloader_cfg and dataloader_cfg.get('dataset'):
                dataset_cfg = dataloader_cfg.dataset
                break
        if dataset_cfg is None:
            raise ValueError('Failed to find any dataset config to infer an export image.')

        ann_file = dataset_cfg.get('ann_file')
        if not ann_file:
            raise ValueError('Dataset config has no ann_file, please pass --img explicitly.')

        ann_path = resolve_config_relative_path(ann_file, model_cfg_path)
        if not ann_path.exists():
            raise FileNotFoundError(f'Annotation file not found: {ann_path}')

        with ann_path.open('r', encoding='utf-8') as handle:
            coco = json.load(handle)

        images = coco.get('images', [])
        if not images:
            raise ValueError(f'No images found in annotation file: {ann_path}')

        file_name = images[0]['file_name']
        data_prefix = dataset_cfg.get('data_prefix', {})
        img_root = data_prefix.get('img', '') if isinstance(data_prefix, dict) else data_prefix

        image_path = Path(file_name)
        if not image_path.is_absolute():
            if img_root:
                image_path = Path(img_root) / file_name
            elif dataset_cfg.get('data_root'):
                image_path = Path(dataset_cfg.get('data_root')) / file_name

        if not image_path.is_absolute():
            image_path = resolve_config_relative_path(str(image_path), model_cfg_path)
        else:
            image_path = image_path.resolve()

        if not image_path.exists():
            raise FileNotFoundError(f'Export image not found: {image_path}')
        return image_path

    requested_checkpoint_path = resolve_path(checkpoint)
    checkpoint_path = resolve_checkpoint_path(requested_checkpoint_path)
    onnx_path = resolve_path(save)
    if requested_checkpoint_path != checkpoint_path and save == './work_dirs/hrnet_w48_topdown/epoch_300.onnx':
        onnx_path = checkpoint_path.with_suffix('.onnx')

    model_cfg_path = resolve_path(model_config) if model_config else infer_model_config(checkpoint_path)
    model_cfg_for_image = prepare_model_cfg(model_cfg_path)
    image_path = resolve_path(img) if img else infer_image_from_dataset(model_cfg_for_image, model_cfg_path)

    model_cfg = prepare_model_cfg(model_cfg_path)
    input_size = extract_input_size(model_cfg)
    deploy_cfg = mmengine.Config(
        dict(
            ir_config=dict(
                type='onnx',
                input_names=['input'],
                output_names=['output'],
                input_shape=[input_size[0], input_size[1]],
                opset_version=11,
            ),
            codebase_config=dict(type='mmpose', task='PoseDetection'),
            backend_config=dict(
                type='tensorrt',
                common_config=dict(
                    max_workspace_size=1 << 30,
                    fp16_mode=False,
                ),
                model_inputs=[
                    dict(
                        input_shapes=dict(
                            input=dict(
                                min_shape=[1, 3, input_size[1], input_size[0]],
                                opt_shape=[1, 3, input_size[1], input_size[0]],
                                max_shape=[1, 3, input_size[1], input_size[0]],
                            )
                        )
                    )
                ],
            ),
        )
    )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch2onnx(
        str(image_path),
        str(onnx_path.parent),
        onnx_path.name,
        deploy_cfg,
        model_cfg,
        model_checkpoint=str(checkpoint_path),
        device=device,
    )

    if not onnx_path.exists():
        raise FileNotFoundError(f'ONNX export failed, file not found: {onnx_path}')

    if verbose:
        print(f'model_config: {model_cfg_path}')
        print(f'image: {image_path}')
        print(f'input_size: {tuple(input_size)}')
        print(f'onnx: {onnx_path}')

    return onnx_path

if __name__ == '__main__':
    
    # exportYoloOnnx()
    exportHRNetOnnx()