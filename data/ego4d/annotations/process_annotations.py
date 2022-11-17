'''
process Ego4D official annotations to our annotation format
'''
import argparse
import json


def process(args):
    ######################################################################################################
    #                     Load data annotations
    ######################################################################################################
    annotation_path = args.official_dir
    info_path = annotation_path + 'ego4d.json'
    annot_path_train = annotation_path + 'nlq_train.json'
    annot_path_val = annotation_path + 'nlq_val.json'
    annot_path_test = annotation_path + 'nlq_test_unannotated.json'

    with open(annot_path_train, 'r') as f:
        v_annot_train = json.load(f)

    with open(annot_path_val, 'r') as f:
        v_annot_val = json.load(f)

    with open(annot_path_test, 'r') as f:
        v_annot_test = json.load(f)

    with open(info_path, 'r') as f:
        feat_info = json.load(f)

    v_all_duration = {}
    for video in feat_info['videos']:
        v_id = video['video_uid']
        v_dur = video['duration_sec']
        v_all_duration[v_id] = v_dur

    v_annot = {}
    v_annot['videos'] = v_annot_train['videos'] + v_annot_val['videos'] + v_annot_test['videos']
    all_videos = []

    ######################################################################################################
    #                     Convert video annotations to clip annotations: clip_annot_1
    ######################################################################################################
    clip_annot_1 = {}
    for video in v_annot['videos']:
        vid = video['video_uid']

        if vid not in all_videos:
            all_videos.append(vid)

        clips = video['clips']
        v_duration = v_all_duration[vid]
        for clip in clips:
            clip_id = clip['clip_uid']

            if clip_id not in clip_annot_1.keys():
                clip_annot_1[clip_id] = {}
                clip_annot_1[clip_id]['video_id'] = vid
                clip_annot_1[clip_id]['clip_id'] = clip_id
                clip_annot_1[clip_id]['duration'] = int(clip['clip_end_sec'] - clip['clip_start_sec'])
                clip_annot_1[clip_id]['fps'] = 30
                clip_annot_1[clip_id]['annotations'] = []
                clip_annot_1[clip_id]['subset'] = video['split']
                clip_annot_1[clip_id]['num_frames'] = int((clip['clip_end_sec'] - clip['clip_start_sec'])*30/16)

            # if video['split'] != 'test':
            annotations = clip['annotations']
#            anno_uid = annot['annotation_uid']
            for cnt, annot in enumerate(annotations):
                idx = 0

                queries = annot['language_queries']
                anno_uid = annot['annotation_uid']
                # print(queries)
                for query in queries:

                    content = query
                    content1 = dict()
                    try:
                        content1 = {'sentence': content['query'].lower(),
                                    'segment': [content['clip_start_sec'], content['clip_end_sec']],
                                    'annotation_uid': anno_uid, 'query_idx': idx}
                    except:
                        if video['split'] == 'train' or video['split'] == 'val':
                            print("Missing: ")
                            print(clip_id)
                            print(query)
                            print("\n")
                            continue
                        else:
                            try:
                                content1 = {'sentence': content['query'].lower(),
                                            'segment': [0, 0],
                                            'annotation_uid': anno_uid, 'query_idx': idx}
                            except:
                                print("Missing: ")
                                print("test")
                                print(clip_id)
                                print(query)
                                print("\n")
                                exit()
                    idx += 1
                    clip_annot_1[clip_id]['annotations'].append(content1)

    train = {}
    val = {}
    test = {}

    # split to 3 set:
    for key in clip_annot_1.keys():
        if clip_annot_1[key]['subset'] == 'train':
            train.update({key: clip_annot_1[key]})
        elif clip_annot_1[key]['subset'] == 'val':
            val.update({key: clip_annot_1[key]})
        else:
            test.update({key: clip_annot_1[key]})

    all_con = {'train': train, 'val': val, 'test': test}

    for key in all_con.keys():
        print("Split %s: %d video clips" % (str(key), len(all_con[key].keys())))

    with open("ego4d.json", "w") as fp:
        json.dump(all_con, fp)

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', "--official_dir", required=True, default="../official_anno/",
                        help="Path to Ego4d official annotation file folder")
    # Change to your own path containing canonical annotation files
    args = parser.parse_args()
    process(args)
