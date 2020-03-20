"""
This script is to generate mixture speech with noise
The code is based on the code of Pandey
Author: Andong Li
Time: 2019/12/15
"""
import argparse
import random
import sys
import os
import json
import time
import h5py
import numpy as np
import soundfile as sf


def gen_train_mix(args):

    print('Begin to generate mix utterances for train dataset')
    ##  read and write path
    train_speech_path = os.path.join(args.speech_path, 'train')
    train_noise_path = os.path.join(args.noise_path, 'seen noise')
    train_save_path = os.path.join(args.mixture_path, 'train')
    os.makedirs(train_save_path, exist_ok= True)
    train_raw_clean_path = os.path.join(train_save_path, 'raw clean')
    os.makedirs(train_raw_clean_path, exist_ok= True)
    train_raw_mix_path = os.path.join(train_save_path, 'raw mix')
    os.makedirs(train_raw_mix_path, exist_ok= True)
    train_json_path = os.path.join(args.json_path, 'train')
    os.makedirs(train_json_path, exist_ok= True)


    # parameter configurations
    fs = args.fs
    train_snr_list = args.train_snr_list
    mix_num = args.train_mix_num
    train_noise = 'seen_long_noise.bin'
    speech_list = os.listdir(train_speech_path)
    speech_list.sort()

    mix_json = []
    clean_json = []

    # read noise bin
    n = np.memmap(os.path.join(train_noise_path, train_noise), dtype = np.float32, mode = 'r')
    for count in range(mix_num):
        random.seed(time.time())
        s_c = random.randint(0, len(speech_list)-1)
        snr_c = random.randint(0, len(train_snr_list)- 1)

        speech_name = speech_list[s_c]
        s, s_fs = sf.read(os.path.join(train_speech_path, speech_name))
        if s_fs != fs:
            raise ValueError('Invalid sample rate!')
        snr = train_snr_list[snr_c]

        ## choose a point to cut the noise
        # if n.size >= s.size:
        #     noise_begin = random.randint(0, n.size - s.size)
        #     n_t = n[noise_begin: noise_begin + s.size]
        # else:
        #     rep = np.ceil(s.size/n.size)
        #     n_t = n.repeat(rep)
        #     n_t = n_t[0:len(s)]

        noise_begin = random.randint(0, n.size - s.size)
        while np.sum(n[noise_begin:noise_begin+ s.size]** 2.0) == 0.0:
            noise_begin = random.randint(0, n.size - s.size)
        n_t = n[noise_begin: noise_begin + s.size]
        alpha = np.sqrt(np.sum(s ** 2.0) / (np.sum(n_t ** 2.0) * (10.0 ** (snr / 10.0))))
        snr_check = 10.0 * np.log10(np.sum(s ** 2.0) / (np.sum((n_t * alpha) ** 2.0)))
        mix = s + alpha * n_t

        # save the file with dat file format
        file_name = os.path.splitext(speech_name)[0]
        snr_name = "%s" % (snr)
        mix_file_name = "%s_%s_id_%s_mix.wav" % (file_name, snr_name, count+1)
        clean_file_name = "%s_%s_id_%s_clean.wav" % (file_name, snr_name, count+1)
        mix_file_path = os.path.join(train_raw_mix_path, mix_file_name)
        clean_file_path = os.path.join(train_raw_clean_path, clean_file_name)

        mix_json.append(mix_file_path)
        clean_json.append(clean_file_path)

        # write wav file
        sf.write(mix_file_path, mix, fs)
        sf.write(clean_file_path, s, fs)
        print('Speech index %s has been generated' %(count+ 1))

    # save json file
    with open(os.path.join(train_json_path, 'mix.json') ,'w') as f :
        json.dump(mix_json, f, indent= 4)
    with open(os.path.join(train_json_path, 'clean.json'), 'w') as f:
        json.dump(clean_json, f, indent= 4)


def gen_cv_mix(args):

    print('Begin to generate mix utterances for cv dataset')
    ##  read and write path
    cv_speech_path = os.path.join(args.speech_path, 'cv')
    cv_noise_path = os.path.join(args.noise_path, 'seen noise')
    cv_save_path = os.path.join(args.mixture_path, 'cv')
    os.makedirs(cv_save_path, exist_ok= True)
    cv_raw_clean_path = os.path.join(cv_save_path, 'raw clean')
    os.makedirs(cv_raw_clean_path, exist_ok= True)
    cv_raw_mix_path = os.path.join(cv_save_path, 'raw mix')
    os.makedirs(cv_raw_mix_path, exist_ok= True)
    cv_json_path = os.path.join(args.json_path, 'cv')
    os.makedirs(cv_json_path, exist_ok= True)


    # parameter configurations
    fs = args.fs
    cv_snr_list = args.train_snr_list
    mix_num = args.cv_mix_num
    cv_noise = 'seen_long_noise.bin'
    speech_list = os.listdir(cv_speech_path)
    speech_list.sort()

    mix_json = []
    clean_json = []

    # read noise bin
    n = np.memmap(os.path.join(cv_noise_path, cv_noise), dtype = np.float32, mode = 'r')
    for count in range(mix_num):
        random.seed(time.time())
        s_c = random.randint(0, len(speech_list)-1)
        snr_c = random.randint(0, len(cv_snr_list)- 1)

        speech_name = speech_list[s_c]
        s, s_fs = sf.read(os.path.join(cv_speech_path, speech_name))
        if s_fs != fs:
            raise ValueError('Invalid sample rate!')
        snr = cv_snr_list[snr_c]

        ## choose a point to cut the noise
        # if n.size >= s.size:
        #     noise_begin = random.randint(0, n.size - s.size)
        #     n_t = n[noise_begin: noise_begin + s.size]
        # else:
        #     rep = np.ceil(s.size/n.size)
        #     n_t = n.repeat(rep)
        #     n_t = n_t[0:len(s)]

        noise_begin = random.randint(0, n.size - s.size)
        while np.sum(n[noise_begin:noise_begin+ s.size]** 2.0) == 0.0:
            noise_begin = random.randint(0, n.size - s.size)
        n_t = n[noise_begin: noise_begin + s.size]
        alpha = np.sqrt(np.sum(s ** 2.0) / (np.sum(n_t ** 2.0) * (10.0 ** (snr / 10.0))))
        snr_check = 10.0 * np.log10(np.sum(s ** 2.0) / (np.sum((n_t * alpha) ** 2.0)))
        mix = s + alpha * n_t

        # save the file with dat file format
        file_name = os.path.splitext(speech_name)[0]
        snr_name = "%s" % (snr)
        mix_file_name = "%s_%s_id_%s_mix.wav" % (file_name, snr_name, count + 1)
        clean_file_name = "%s_%s_id_%s_clean.wav" % (file_name, snr_name, count + 1)
        mix_file_path = os.path.join(cv_raw_mix_path, mix_file_name)
        clean_file_path = os.path.join(cv_raw_clean_path, clean_file_name)

        mix_json.append(mix_file_path)
        clean_json.append(clean_file_path)

        # write wav file
        sf.write(mix_file_path, mix, fs)
        sf.write(clean_file_path, s, fs)
        print('Speech index %s has been generated' %(count+ 1))

    # save json file
    with open(os.path.join(cv_json_path, 'mix.json') ,'w') as f :
        json.dump(mix_json, f, indent= 4)
    with open(os.path.join(cv_json_path, 'clean.json'), 'w') as f:
        json.dump(clean_json, f, indent= 4)


def gen_seen_test_mix(args):

    print('Begin to generate mix utterances for seen test dataset')
    ##  read and write path
    test_speech_path = os.path.join(args.speech_path, 'test')
    test_noise_path = os.path.join(args.noise_path, 'seen noise')
    test_save_path = os.path.join(args.mixture_path, 'seen_test')
    os.makedirs(test_save_path, exist_ok= True)
    test_raw_clean_path = os.path.join(test_save_path, 'raw clean')
    os.makedirs(test_raw_clean_path, exist_ok= True)
    test_raw_mix_path = os.path.join(test_save_path, 'raw mix')
    os.makedirs(test_raw_mix_path, exist_ok= True)
    test_json_path = os.path.join(args.json_path, 'seen_test')
    os.makedirs(test_json_path, exist_ok= True)

    # parameter configurations
    fs = args.fs
    test_snr_list = args.test_snr_list
    mix_num = args.test_mix_num
    test_noise = 'seen_long_noise.bin'
    speech_list = os.listdir(test_speech_path)
    speech_list.sort()

    mix_json = []
    clean_json = []

    # read noise bin
    n = np.memmap(os.path.join(test_noise_path, test_noise), dtype = np.float32, mode = 'r')
    for count in range(mix_num):
        random.seed(time.time())
        s_c = random.randint(0, len(speech_list)-1)
        snr_c = random.randint(0, len(test_snr_list)- 1)

        speech_name = speech_list[s_c]
        s, s_fs = sf.read(os.path.join(test_speech_path, speech_name))
        if s_fs != fs:
            raise ValueError('Invalid sample rate!')
        snr = test_snr_list[snr_c]

        ## choose a point to cut the noise
        # if n.size >= s.size:
        #     noise_begin = random.randint(0, n.size - s.size)
        #     n_t = n[noise_begin: noise_begin + s.size]
        # else:
        #     rep = np.ceil(s.size/n.size)
        #     n_t = n.repeat(rep)
        #     n_t = n_t[0:len(s)]

        noise_begin = random.randint(0, n.size - s.size)
        while np.sum(n[noise_begin:noise_begin+ s.size]** 2.0) == 0.0:
            noise_begin = random.randint(0, n.size - s.size)
        n_t = n[noise_begin: noise_begin + s.size]
        alpha = np.sqrt(np.sum(s ** 2.0) / (np.sum(n_t ** 2.0) * (10.0 ** (snr / 10.0))))
        snr_check = 10.0 * np.log10(np.sum(s ** 2.0) / (np.sum((n_t * alpha) ** 2.0)))
        mix = s + alpha * n_t

        # save the file with dat file format
        file_name = os.path.splitext(speech_name)[0]
        snr_name = "%s" % (snr)
        mix_file_name = "%s_%s_id_%s_mix.wav" % (file_name, snr_name, count + 1)
        clean_file_name = "%s_%s_id_%s_clean.wav" % (file_name, snr_name, count + 1)
        mix_file_path = os.path.join(test_raw_mix_path, mix_file_name)
        clean_file_path = os.path.join(test_raw_clean_path, clean_file_name)

        mix_json.append(mix_file_path)
        clean_json.append(clean_file_path)

        # write wav file
        sf.write(mix_file_path, mix, fs)
        sf.write(clean_file_path, s, fs)
        print('Speech index %s has been generated' %(count+ 1))

    # save json file
    with open(os.path.join(test_json_path, 'mix.json'), 'w') as f :
        json.dump(mix_json, f, indent= 4)
    with open(os.path.join(test_json_path, 'clean.json'), 'w') as f:
        json.dump(clean_json, f, indent= 4)


def gen_unseen_test_mix(args):

    print('Begin to generate mix utterances for unseen test dataset')
    ##  read and write path
    test_speech_path = os.path.join(args.speech_path, 'test')
    test_noise_path = os.path.join(args.noise_path, 'unseen noise')
    test_save_path = os.path.join(args.mixture_path, 'unseen_test')
    os.makedirs(test_save_path, exist_ok= True)
    test_raw_clean_path = os.path.join(test_save_path, 'raw clean')
    os.makedirs(test_raw_clean_path, exist_ok= True)
    test_raw_mix_path = os.path.join(test_save_path, 'raw mix')
    os.makedirs(test_raw_mix_path, exist_ok= True)
    test_json_path = os.path.join(args.json_path, 'unseen_test')
    os.makedirs(test_json_path, exist_ok= True)

    # parameter configurations
    fs = args.fs
    test_snr_list = args.test_snr_list
    mix_num = args.test_mix_num
    test_noise = 'unseen_long_noise.bin'
    speech_list = os.listdir(test_speech_path)
    speech_list.sort()

    mix_json = []
    clean_json = []

    # read noise bin
    n = np.memmap(os.path.join(test_noise_path, test_noise), dtype = np.float32, mode = 'r')
    for count in range(mix_num):
        random.seed(time.time())
        s_c = random.randint(0, len(speech_list)-1)
        snr_c = random.randint(0, len(test_snr_list)- 1)

        speech_name = speech_list[s_c]
        s, s_fs = sf.read(os.path.join(test_speech_path, speech_name))
        if s_fs != fs:
            raise ValueError('Invalid sample rate!')
        snr = test_snr_list[snr_c]

        ## choose a point to cut the noise
        # if n.size >= s.size:
        #     noise_begin = random.randint(0, n.size - s.size)
        #     n_t = n[noise_begin: noise_begin + s.size]
        # else:
        #     rep = np.ceil(s.size/n.size)
        #     n_t = n.repeat(rep)
        #     n_t = n_t[0:len(s)]

        noise_begin = random.randint(0, n.size - s.size)
        while np.sum(n[noise_begin:noise_begin+ s.size]** 2.0) == 0.0:
            noise_begin = random.randint(0, n.size - s.size)
        n_t = n[noise_begin: noise_begin + s.size]
        alpha = np.sqrt(np.sum(s ** 2.0) / (np.sum(n_t ** 2.0) * (10.0 ** (snr / 10.0))))
        snr_check = 10.0 * np.log10(np.sum(s ** 2.0) / (np.sum((n_t * alpha) ** 2.0)))
        mix = s + alpha * n_t

        # save the file with dat file format
        file_name = os.path.splitext(speech_name)[0]
        snr_name = "%s" % (snr)
        mix_file_name = "%s_%s_id_%s_mix.wav" % (file_name, snr_name, count + 1)
        clean_file_name = "%s_%s_id_%s_clean.wav" % (file_name, snr_name, count + 1)
        mix_file_path = os.path.join(test_raw_mix_path, mix_file_name)
        clean_file_path = os.path.join(test_raw_clean_path, clean_file_name)

        mix_json.append(mix_file_path)
        clean_json.append(clean_file_path)

        # write wav file
        sf.write(mix_file_path, mix, fs)
        sf.write(clean_file_path, s, fs)
        print('Speech index %s has been generated' %(count+ 1))

    # save json file
    with open(os.path.join(test_json_path, 'mix.json'), 'w') as f :
        json.dump(mix_json, f, indent= 4)
    with open(os.path.join(test_json_path, 'clean.json'), 'w') as f:
        json.dump(clean_json, f, indent= 4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Timit dataset, to generate the mix dataset")
    parser.add_argument('--speech_path', type= str, default= "/media/Dataset/STIMIT/clean",
                        help= 'The path to the clean utterances from TIMIT dataset')
    parser.add_argument('--mixture_path', type = str, default= "/media/Dataset/STIMIT/mixture",
                        help = 'The path to the incomming mixture utterances')
    parser.add_argument('--noise_path', type = str, default = "/media/Dataset/STIMIT/noise",
                        help = 'The path to the noise signals')
    parser.add_argument('--json_path', type = str, default = "/media/Dataset/STIMIT/json",
                        help = 'The path to generate json files')
    parser.add_argument('--fs', type = int, default= 16000,
                        help = 'sampling rate')
    parser.add_argument('--train_snr_list', type = list, default = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        help = "snr list for train dataset")
    parser.add_argument('--test_snr_list', type=list, default=[-5, -2],
                        help="snr list for test dataset")
    parser.add_argument('--train_mix_num', type = int, default= 10000,
                        help = "the times for snr mixing in train case")
    parser.add_argument('--cv_mix_num', type=int, default = 2000,
                        help="the times for snr mixing in cv case")
    parser.add_argument('--test_mix_num', type=int, default= 400,
                        help="the times for snr mixing in test case")

    gen_args = parser.parse_args()
    print(gen_args)
    gen_train_mix(gen_args)
    gen_cv_mix(gen_args)
    gen_seen_test_mix(gen_args)
    gen_unseen_test_mix(gen_args)