python -m tools.validate \
    --crnn_model_dir=model \
    --image_dir=data/test \
    --charset=charset6k.txt \
    --label_file=data/test.txt \
    --debug=False \
    --validate_num=1000 \
    --crnn_model_file=crnn_2019-04-19-11-42-50.ckpt-0

