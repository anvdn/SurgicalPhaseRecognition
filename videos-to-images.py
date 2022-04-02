import argparse
import multiprocessing as mp
import os
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resize', type=tuple, default=(224, 224), metavar='RES', help="Size of the resized frame")
    args = parser.parse_args()

    # create subfolder to store frames if it does not already exist
    if not os.path.exists(utils.images_path): os.makedirs(utils.images_path)

    # recover names of all videos
    video_names = utils.get_train_test_video_names()
    all_video_names = video_names['train'] + video_names['test']
    
    print("Number of processors: ", mp.cpu_count())
    print("Number of videos to process", len(all_video_names))

    pool = mp.Pool(mp.cpu_count())
    # save resized frames of all videos
    for i in range(len(all_video_names)):
        pool.apply_async(utils.save_frames, args=(all_video_names[i], args.resize))
    pool.close()
    pool.join()