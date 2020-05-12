mo_tf_path='~/intel/openvino/deployment_tools/model_optimizer/mo_tf.py'
#pb_file='/home/vietnguyen/new_deephar/output/penn.pb'
#output_dir='/home/vietnguyen/new_deephar/output/'
pb_file='/home/nbviet/Documents/new_deephar/output/penn.pb'
output_dir='/home/nbviet/Documents/new_deephar/output/'
input_shape=[1,8,256,256,3]

python ${mo_tf_path} --input_model ${pb_file} --output_dir ${output_dir} --input_shape ${input_shape} --data_type FP32
