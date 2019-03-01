#    parser.add_argument('-d', '--dataset_dir', type=str, required=True,
#                        help='Path to dir containing train/test data and annotation files.')
#    parser.add_argument('-w', '--weights_path', type=str, help='Path to pre-trained weights.')
#    parser.add_argument('-j', '--num_threads', type=int, default=int(os.cpu_count()/2),
#                        help='Number of threads to use in batch shuffling')

python \
    -m tools.train_shadownet \
    --dataset_dir=data
#    --weights_path=model/shadownet/shadownet_2017-10-17-11-47-46.ckpt-199999

