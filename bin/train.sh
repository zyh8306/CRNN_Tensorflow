# 第一句表明使用第0个GPU，CRNN用第0个GPU，CTPN用第1个GPU，恩，我这么分配的
CUDA_VISIBLE_DEVICES=0 \
python \
    -m tools.train_shadownet \
    --validate_steps=10 \
    --tboard_dir=tboard \
    --debug=True
#    >> ./logs/crnn.log 2>&1
#    --weights_path=model/shadownet/shadownet_2019-03-11-15-13-38.ckpt-8 \

