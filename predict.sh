
python predict.py --crop_height=624 \
                  --crop_width=960 \
                  --max_disp=48 \
                  --data_path='/robodata/srabiee/AirSim_IVSLAM/cityenv_wb/03010/' \
                  --test_list='lists/airsim_neighborhood_tmp.list' \
                  --save_path='/robodata/user_data/srabiee/results/ipr/depth_prediction/tmp/city_wb_03010/img_disp_pred/' \
                  --airsim_fmt=1 \
                  --resume='pretrained_models/kitti2015_final.pth'
exit      

# python predict.py --crop_height=384 \
#                   --crop_width=1248 \
#                   --max_disp=192 \
#                   --data_path='/ssd1/zhangfeihu/data/kitti/2015//testing/' \
#                   --test_list='lists/kitti2015_test.list' \
#                   --save_path='./result/' \
#                   --kitti2015=1 \
#                   --resume='./checkpoint/kitti2015_final.pth'
# exit

# python predict.py --crop_height=384 \
#                   --crop_width=1248 \
#                   --max_disp=192 \
#                   --data_path='/media/feihu/Storage/stereo/kitti/testing/' \
#                   --test_list='lists/kitti2012_test.list' \
#                   --save_path='./result/' \
#                   --kitti=1 \
#                   --resume='./checkpoint/kitti2012_final.pth'



