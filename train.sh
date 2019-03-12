#    parser.add_argument('-d', '--dataset_dir', type=str, required=True,
#                        help='Path to dir containing train/test data and annotation files.')
#    parser.add_argument('-w', '--weights_path', type=str, help='Path to pre-trained weights.')
#    parser.add_argument('-j', '--num_threads', type=int, default=int(os.cpu_count()/2),
#                        help='Number of threads to use in batch shuffling')
# 第一句表明使用第0个GPU，CRNN用第0个GPU，CTPN用第1个GPU，恩，我这么分配的
CUDA_VISIBLE_DEVICES=0 \
python \
    -m tools.train_shadownet \
    --dataset_dir=data \
    --weights_path=model/shadownet/shadownet_2019-03-11-15-13-38.ckpt-8 \
    >> ./logs/crnn.log 2>&1

