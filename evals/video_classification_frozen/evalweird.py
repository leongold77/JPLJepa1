import os
import logging
import pprint
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import yaml

from src.models.vision_transformer import VisionTransformer
from src.models.attentive_pooler import AttentiveClassifier
from src.datasets.data_manager import init_data
from src.utils.distributed import init_distributed, AllReduce
from src.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule
from src.utils.logging import AverageMeter, CSVLogger
from evals.video_classification_frozen.utils import make_transforms, ClipAggregation, FrameAggregation

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)

def get_vit_model(model_name, **kwargs):
    model_dict = {
        'vit_small': VisionTransformer(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., qkv_bias=True, norm_layer=torch.nn.LayerNorm, **kwargs),
        'vit_base': VisionTransformer(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, norm_layer=torch.nn.LayerNorm, **kwargs),
        'vit_large': VisionTransformer(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4., qkv_bias=True, norm_layer=torch.nn.LayerNorm, **kwargs),
    }
    return model_dict.get(model_name, None)

def main(args_eval, resume_preempt=False):
    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # Load model
    pretrained_path = os.path.join(args_eval['pretrain']['folder'], args_eval['pretrain']['checkpoint'])
    model_name = args_eval['pretrain']['model_name']
    logger.info(f'Model name from config: {model_name}')

    encoder = init_model(
        crop_size=args_eval['optimization']['resolution'],
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=args_eval['pretrain']['patch_size'],
        tubelet_size=args_eval['pretrain']['tubelet_size'],
        frames_per_clip=args_eval['pretrain']['frames_per_clip'],
        uniform_power=args_eval['pretrain']['uniform_power'],
        checkpoint_key=args_eval['pretrain']['checkpoint_key'],
        use_SiLU=args_eval['pretrain']['use_silu'],
        tight_SiLU=args_eval['pretrain']['tight_silu'],
        use_sdpa=args_eval['pretrain']['use_sdpa']
    )
    
    if args_eval['pretrain']['frames_per_clip'] == 1:
        encoder = FrameAggregation(encoder).to(device)
    else:
        encoder = ClipAggregation(
            encoder,
            tubelet_size=args_eval['pretrain']['tubelet_size'],
            attend_across_segments=args_eval['optimization']['attend_across_segments']
        ).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Load attentive probe
    attentive_probe_path = '/home/leon-gold/Downloads/k400-probe.pth.tar'  # Update this path
    classifier = load_attentive_probe(attentive_probe_path, encoder.embed_dim, encoder.num_heads, args_eval['data']['num_classes'])
    classifier.to(device)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    val_loader = make_dataloader(
        dataset_type=args_eval['data']['dataset_type'],
        root_path=args_eval['data']['dataset_val'],
        resolution=args_eval['optimization']['resolution'],
        frames_per_clip=args_eval['data']['frames_per_clip'],
        frame_step=args_eval['pretrain'].get('frame_step', 4),  # Use a default value of 4 if 'frame_step' is missing
        num_segments=args_eval['data']['num_segments'],
        eval_duration=args_eval['pretrain']['clip_duration'],
        num_views_per_segment=args_eval['data']['num_views_per_segment'],
        allow_segment_overlap=True,
        batch_size=1,
        world_size=world_size,
        rank=rank,
        training=False
    )

    res = run_one_epoch(
        device=device,
        training=False,
        num_temporal_views=args_eval['data']['num_segments'],
        attend_across_segments=args_eval['optimization']['attend_across_segments'],
        num_spatial_views=args_eval['data']['num_views_per_segment'],
        encoder=encoder,
        classifier=classifier,
        scaler=None,
        optimizer=None,
        scheduler=None,
        wd_scheduler=None,
        data_loader=val_loader,
        use_bfloat16=args_eval['optimization']['use_bfloat16']
    )

    print("Evaluation results:", res)


def init_model(
    device,
    pretrained,
    model_name,
    patch_size=16,
    crop_size=224,
    frames_per_clip=16,
    tubelet_size=2,
    use_sdpa=False,
    use_SiLU=False,
    tight_SiLU=True,
    uniform_power=False,
    checkpoint_key='target_encoder'
):
    logger.info(f'Initializing model: {model_name}')  # Log the model name for debugging
    encoder = get_vit_model(
        model_name,
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
    )
    if encoder is None:
        raise ValueError(f'Model name {model_name} is not recognized.')
    encoder.to(device)
    encoder = load_pretrained(encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key)
    return encoder


def load_pretrained(encoder, pretrained, checkpoint_key='target_encoder'):
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    pretrained_dict = checkpoint[checkpoint_key]
    # Strip the 'backbone.' prefix from keys
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()} # also handling module prefix
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f'loaded pretrained model with msg: {msg}')
    logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    return encoder


def load_attentive_probe(path, embed_dim, num_heads, num_classes):
    logger.info(f'Loading attentive probe from {path}')
    classifier = AttentiveClassifier(
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=1,
        num_classes=num_classes,
    )
    checkpoint = torch.load(path, map_location='cpu')
    logger.info(f'Checkpoint keys: {list(checkpoint.keys())}')  # Print the keys in the checkpoint
    # Strip the 'module.' prefix from keys
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['classifier'].items()}
    classifier.load_state_dict(state_dict)
    return classifier


def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type='VideoDataset',
    resolution=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=2,
    subset_file=None
):
    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4/3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=resolution,
    )

def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type='VideoDataset',
    resolution=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    subset_file=None
):
    
    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4/3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=resolution,
    )

    data_loader, _ = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_segments,
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        copy_data=False,
        drop_last=False,
        subset_file=subset_file)
    return data_loader

    # Debugging statement to check the data loader
    for i, data in enumerate(data_loader):
        print(f'Loaded data shapes at batch {i}: {[d.shape for d in data[0]]}, Labels shape: {data[1].shape}')
        if i >= 5:  # Only print the first 5 batches for brevity
            break

    return data_loader

    # Debugging statement to check the data loader
    for data in data_loader:
        print(f'Loaded data shapes: {[d.shape for d in data[0]]}, Labels shape: {data[1].shape}')
        break

    return data_loader


def run_one_epoch(
    device,
    training,
    encoder,
    classifier,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    num_spatial_views,
    num_temporal_views,
    attend_across_segments,
):
    classifier.train(mode=training)
    criterion = torch.nn.CrossEntropyLoss()
    top1_meter = AverageMeter()

    global_idx = 0  # Initialize a global index

    for itr, data in enumerate(data_loader):
        if training:
            scheduler.step()
            wd_scheduler.step()

        clips = [
            [dij.to(device, non_blocking=True).float() for dij in di]
            for di in data[0]
        ]
        clip_indices = [d.to(device, non_blocking=True).float() for d in data[2]]
        labels = data[1].to(device).long()  # Ensure labels are int64
        video_paths = data[2]  # Video paths for debugging
        video_order = list(range(global_idx, global_idx + len(labels)))  # Generate global indices for the batch

        # Print the current batch video paths and labels for verification
        print(f"Batch {itr} - Video order: {video_order}")
        print(f"Labels: {labels.tolist()}")
        #print(f"Video Paths: {video_paths}")

        batch_size = len(labels)

        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_bfloat16):
            outputs = encoder(clips, clip_indices)

        if attend_across_segments:
            outputs = [classifier(o).float() for o in outputs]
        else:
            outputs = [[classifier(ost).float() for ost in os] for os in outputs]

        loss = 0
        if attend_across_segments:
            for o in outputs:
                loss += criterion(o, labels)
            loss /= len(outputs)
        else:
            for os in outputs:
                for ost in os:
                    loss += criterion(ost, labels)
            loss /= len(outputs) * len(outputs[0])

        if training:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            if attend_across_segments:
                outputs = [F.softmax(o, dim=1) for o in outputs]
                predicted_classes = [torch.argmax(o, dim=1) for o in outputs]
                confidences = [torch.max(o, dim=1)[0] for o in outputs]
            else:
                outputs = [[F.softmax(ost, dim=1) for ost in os] for os in outputs]
                predicted_classes = [[torch.argmax(ost, dim=1) for ost in os] for os in outputs]
                confidences = [[torch.max(ost, dim=1)[0] for ost in os] for os in outputs]

            # Print video indices, predicted classes, and confidence intervals
            if attend_across_segments:
                for batch_idx, (batch_pred_classes, batch_confidences) in enumerate(zip(predicted_classes, confidences)):
                    print(f"Batch {batch_idx}:")
                    for sample_idx, (pred_class, confidence) in enumerate(zip(batch_pred_classes, batch_confidences)):
                        video_idx = video_order[sample_idx]
                        print(f"  Video index: {video_idx} - Predicted class = {pred_class.item()}, Confidence = {confidence.item():.4f}")
            else:
                for temporal_idx, (temporal_pred_classes, temporal_confidences) in enumerate(zip(predicted_classes, confidences)):
                    print(f"  Temporal Segment {temporal_idx}:")
                    for sample_idx, (pred_class, confidence) in enumerate(zip(temporal_pred_classes, temporal_confidences)):
                        video_idx = video_order[sample_idx]
                        print(f"    Video index: {video_idx} - Predicted class = {pred_class.item()}, Confidence = {confidence.item():.4f}")

        if attend_across_segments:
            top1_acc = 0
            for o in outputs:
                top1_acc += 100. * o.max(dim=1).indices.eq(labels).sum().item() / batch_size
            top1_acc /= len(outputs)
        else:
            top1_acc = 0
            for os in outputs:
                for ost in os:
                    top1_acc += 100. * ost.max(dim=1).indices.eq(labels).sum().item() / batch_size
            top1_acc /= len(outputs) * len(outputs[0])
        top1_meter.update(top1_acc)

        global_idx += batch_size  # Update global index

        if itr % 20 == 0:
            logger.info('[%5d] %.3f%% (loss: %.3f) [mem: %.2e]'
                        % (itr, top1_meter.avg, loss,
                           torch.cuda.max_memory_allocated() / 1024.**2))

    return top1_meter.avg

# Call the training function



    


if __name__ == '__main__':
    # Load the YAML configuration file
    with open('/configs/evals/vitl16_k400_16x8x3.yaml', 'r') as y_file:
        args_eval = yaml.load(y_file, Loader=yaml.FullLoader)
    main(args_eval)
