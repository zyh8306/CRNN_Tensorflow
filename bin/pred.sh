echo "帮助：\n\t如果不指定文件名，识别data/test/目录下的所有图片，否则具体的照片"
echo "\t如果指定model名字，就加载，否则，加载最新的模型名字"
python -m tools.pred \
    --crnn_model_dir=model \
    --image_dir=data/test \
    --charset=charset.txt \
    --image_file= \
    --debug=False \
    --crnn_model_file=crnn_2019-04-03-06-55-13.ckpt-199000

