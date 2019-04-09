if [ "$1" = "" ]; then
    echo "Usage: pred.sh <image_name>  default directory is data/test/"
    exit
fi

python -m tools.pred \
    --model_dir=model \
    --image_path=data/test/$1 \
    --debug=True
#    --weights_path=model/shadownet/shadownet_2019-03-15-16-23-16.ckpt-380

