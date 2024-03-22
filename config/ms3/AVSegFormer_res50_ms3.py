model = dict(
    type='AVSegFormer',
    neck=None,
    backbone=dict(
        type='res50',
        init_weights_path='pretrained/resnet50-19c8e357.pth'),
    vggish=dict(
        freeze_audio_extractor=True,
        pretrained_vggish_model_path='pretrained/vggish-10086976.pth',
        preprocess_audio_to_log_mel=False,
        postprocess_log_mel_with_pca=False,
        pretrained_pca_params_path=None),
    head=dict(
        type='AVSegHead',
        in_channels=[256, 512, 1024, 2048],
        num_classes=1,
        query_num=300,
        use_learnable_queries=True,
        aux_output=True,
        fusion_block=dict(type='CrossModalMixer'),
        matcher=dict(
            type='HungarianMatcher',
            num_queries=300),
        query_generator=dict(
            type='AttentionGenerator',
            num_layers=6,
            query_num=300),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128),
        transformer=dict(
            type='AVSTransformer',
            encoder=dict(
                num_layers=6,
                layer=dict(
                    dim=256,
                    ffn_dim=2048,
                    dropout=0.1)),
            decoder=dict(
                num_layers=6,
                layer=dict(
                    dim=256,
                    ffn_dim=2048,
                    dropout=0.1)))),
    audio_dim=128,
    embed_dim=256,
    freeze_audio_backbone=True,
    T=5)
dataset = dict(
    train=dict(
        type='MS3Dataset',
        split='train',
        anno_csv='data/Multi-sources/ms3_meta_data.csv',
        dir_img='data/Multi-sources/ms3_data/visual_frames',
        dir_audio_log_mel='data/Multi-sources/ms3_data/audio_log_mel',
        dir_mask='data/Multi-sources/ms3_data/gt_masks',
        img_size=(224, 224),
        batch_size=2),
    val=dict(
        type='MS3Dataset',
        split='val',
        anno_csv='data/Multi-sources/ms3_meta_data.csv',
        dir_img='data/Multi-sources/ms3_data/visual_frames',
        dir_audio_log_mel='data/Multi-sources/ms3_data/audio_log_mel',
        dir_mask='data/Multi-sources/ms3_data/gt_masks',
        img_size=(224, 224),
        batch_size=2),
    test=dict(
        type='MS3Dataset',
        split='test',
        anno_csv='data/Multi-sources/ms3_meta_data.csv',
        dir_img='data/Multi-sources/ms3_data/visual_frames',
        dir_audio_log_mel='data/Multi-sources/ms3_data/audio_log_mel',
        dir_mask='data/Multi-sources/ms3_data/gt_masks',
        img_size=(224, 224),
        batch_size=2))
optimizer = dict(
    type='AdamW',
    lr=2e-5)
loss = dict(
    loss_type=['dice', 'l1', 'mix'],
    weight_dict=[1., 1., 0.1])
process = dict(
    num_works=8,
    train_epochs=60,
    freeze_epochs=10)
