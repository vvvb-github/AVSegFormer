model = dict(
    type='AVSegFormer',
    neck=None,
    backbone=dict(
        type='pvt_v2_b5',
        init_weights_path='pretrained/pvt_v2_b5.pth'),
    vggish=dict(
        freeze_audio_extractor=True,
        pretrained_vggish_model_path='pretrained/vggish-10086976.pth',
        preprocess_audio_to_log_mel=True,
        postprocess_log_mel_with_pca=False,
        pretrained_pca_params_path=None),
    head=dict(
        type='AVSegHead',
        in_channels=[64, 128, 320, 512],
        num_classes=71,
        query_num=300,
        use_learnable_queries=True,
        fusion_block=dict(type='CrossModalMixer'),
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
    T=10)
dataset = dict(
    train=dict(
        type='V2Dataset',
        split='train',
        num_class=71,
        mask_num=10,
        crop_img_and_mask=True,
        crop_size=224,
        meta_csv_path='data/AVSS/metadata.csv',
        label_idx_path='data/AVSS/label2idx.json',
        dir_base='data/AVSS',
        img_size=(224, 224),
        resize_pred_mask=True,
        save_pred_mask_img_size=(360, 240),
        batch_size=4),
    val=dict(
        type='V2Dataset',
        split='val',
        num_class=71,
        mask_num=10,
        crop_img_and_mask=True,
        crop_size=224,
        meta_csv_path='data/AVSS/metadata.csv',
        label_idx_path='data/AVSS/label2idx.json',
        dir_base='data/AVSS',
        img_size=(224, 224),
        resize_pred_mask=True,
        save_pred_mask_img_size=(360, 240),
        batch_size=4),
    test=dict(
        type='V2Dataset',
        split='test',
        num_class=71,
        mask_num=10,
        crop_img_and_mask=True,
        crop_size=224,
        meta_csv_path='data/AVSS/metadata.csv',
        label_idx_path='data/AVSS/label2idx.json',
        dir_base='data/AVSS',
        img_size=(224, 224),
        resize_pred_mask=True,
        save_pred_mask_img_size=(360, 240),
        batch_size=4))
optimizer = dict(
    type='AdamW',
    lr=2e-5)
loss = dict(
    weight_dict=dict(
        iou_loss=1.0,
        mask_loss=0.1))
process = dict(
    num_works=8,
    train_epochs=30,
    start_eval_epoch=10,
    eval_interval=2,
    freeze_epochs=0)
