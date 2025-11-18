import argparse
import functools
import json
import os
import random
import re
from multiprocessing import Pool, cpu_count

import librosa
import numpy as np
import soundfile
from tqdm import tqdm

from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("json_file", default="95598智能客服语音数据集/json_19343_20221009173201.json",
        type=str, help="95598数据集JSON标注文件路径")
add_arg("audio_dir", default="95598智能客服语音数据集/广州局95598/",
        type=str, help="音频文件所在目录")
add_arg("target_dir", default="dataset/95598_v3/", type=str, help="输出标注文件的目录")
add_arg('add_pun', default=False, type=bool, help="是否添加标点符")
add_arg('num_workers', default=None, type=int, help="并行处理的进程数，默认使用CPU核心数")
add_arg('min_duration', default=0.5, type=float, help="过滤过短的语音片段（秒）")
add_arg('max_duration', default=30.0, type=float, help="过滤过长的语音片段（秒）")
add_arg('short_max_duration', default=10.0, type=float, help="短片段最大时长（秒）")
add_arg('long_min_duration', default=30.0, type=float, help="长片段最小时长（秒）")
add_arg('segment_merge_gap', default=3.0, type=float, help="合并相邻片段的最大时间间隔（秒）")
add_arg('use_full_audio', default=True, type=bool, help="是否使用完整音频文件作为训练数据")
add_arg('data_augmentation', default=True, type=bool, help="是否启用数据增强")
add_arg('augmentation_ratio', default=3.0, type=float, help="数据增强倍数")
add_arg('noise_addition_prob', default=0.3, type=float, help="添加噪声的概率")
add_arg('speed_perturb_prob', default=0.3, type=float, help="语速扰动概率")
add_arg('volume_perturb_prob', default=0.3, type=float, help="音量扰动概率")
args = parser.parse_args()


def load_punctuation_model():
    """加载标点符号恢复模型"""
    if args.add_pun:
        import logging

        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        from modelscope.utils.logger import get_logger

        logger = get_logger(log_level=logging.CRITICAL)
        logger.setLevel(logging.CRITICAL)

        inference_pipeline = pipeline(
            task=Tasks.punctuation,
            model='iic/punc_ct-transformer_cn-en-common-vocab471067-large',
            model_revision="v2.0.4"
        )
        return inference_pipeline
    return None


def clean_text(text):
    """清理文本，移除特殊标记"""
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', '', text)
    return text.strip()


def add_noise_to_audio(audio, noise_level=0.01):
    """添加高斯噪声"""
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise


def change_speed(audio, speed_factor):
    """改变音频速度"""
    if speed_factor == 1.0:
        return audio

    # 使用librosa改变速度
    return librosa.effects.time_stretch(audio, rate=speed_factor)


def change_volume(audio, volume_factor):
    """改变音频音量"""
    return audio * volume_factor


def augment_audio(audio, sample_rate):
    """数据增强"""
    if not args.data_augmentation:
        return [audio]

    augmented_audios = [audio]  # 原始音频

    # 添加噪声
    if random.random() < args.noise_addition_prob:
        noise_audio = add_noise_to_audio(audio, noise_level=0.005)
        augmented_audios.append(noise_audio)

    # 语速扰动
    if random.random() < args.speed_perturb_prob:
        speed_factors = [0.9, 0.95, 1.05, 1.1]
        for speed_factor in speed_factors:
            if random.random() < 0.5:  # 50%概率选择每个速度因子
                speed_audio = change_speed(audio, speed_factor)
                augmented_audios.append(speed_audio)

    # 音量扰动
    if random.random() < args.volume_perturb_prob:
        volume_factors = [0.8, 0.9, 1.1, 1.2]
        for volume_factor in volume_factors:
            if random.random() < 0.5:  # 50%概率选择每个音量因子
                volume_audio = change_volume(audio, volume_factor)
                augmented_audios.append(volume_audio)

    return augmented_audios


def merge_segments(segments, max_duration=120.0, max_gap=5.0):
    """合并相邻的语音片段，生成长音频训练样本"""
    if not segments:
        return []

    segments = sorted(segments, key=lambda x: x['sentences'][0]['start'])

    merged_segments = []
    current_merge = {
        'audio': {'path': segments[0]['audio']['path']},
        'sentence': '',
        'duration': 0,
        'sentences': [],
        'speaker': segments[0].get('speaker', 'unknown'),
        'role': segments[0].get('role', 'unknown'),
        'audio_filename': segments[0].get('audio_filename', ''),
        'segment_ids': []
    }

    for segment in segments:
        segment_start = segment['sentences'][0]['start']
        segment_end = segment['sentences'][-1]['end']
        segment_duration = segment_end - segment_start

        can_merge = False
        if not current_merge['sentences']:
            can_merge = True
        else:
            current_end = current_merge['sentences'][-1]['end']
            gap = segment_start - current_end
            if (gap <= max_gap and
                current_merge['duration'] + segment_duration + gap <= max_duration):
                can_merge = True

        if can_merge:
            current_merge['sentences'].extend(segment['sentences'])
            current_merge['sentence'] += segment['sentence']
            current_merge['duration'] += segment_duration + (segment_start - current_merge['sentences'][0]['start'] if current_merge['sentences'] else 0)
            current_merge['segment_ids'].append(segment.get('segment_id', ''))
        else:
            if current_merge['duration'] >= args.long_min_duration:
                if current_merge['sentences']:
                    merged_start = current_merge['sentences'][0]['start']
                    merged_end = current_merge['sentences'][-1]['end']
                    current_merge['duration'] = round(merged_end - merged_start, 2)

                    for sent in current_merge['sentences']:
                        sent['start'] = round(sent['start'] - merged_start, 3)
                        sent['end'] = round(sent['end'] - merged_start, 3)

                merged_segments.append(current_merge)

            current_merge = {
                'audio': {'path': segment['audio']['path']},
                'sentence': segment['sentence'],
                'duration': segment_duration,
                'sentences': segment['sentences'].copy(),
                'speaker': segment.get('speaker', 'unknown'),
                'role': segment.get('role', 'unknown'),
                'audio_filename': segment.get('audio_filename', ''),
                'segment_ids': [segment.get('segment_id', '')]
            }

    if current_merge['duration'] >= args.long_min_duration:
        if current_merge['sentences']:
            merged_start = current_merge['sentences'][0]['start']
            merged_end = current_merge['sentences'][-1]['end']
            current_merge['duration'] = round(merged_end - merged_start, 2)

            for sent in current_merge['sentences']:
                sent['start'] = round(sent['start'] - merged_start, 3)
                sent['end'] = round(sent['end'] - merged_start, 3)

        merged_segments.append(current_merge)

    return merged_segments


def process_full_audio(conversation_data, audio_base_dir, punctuation_model=None):
    """处理完整音频文件作为训练样本"""
    try:
        voice_url = conversation_data['data']['voice_url']
        audio_filename = voice_url.split('/')[-1]
        audio_path = os.path.join(audio_base_dir, audio_filename)

        if not os.path.exists(audio_path):
            print(f"音频文件不存在: {audio_path}")
            return None

        # 读取完整音频
        try:
            sample, sr = soundfile.read(audio_path)
            total_duration = sample.shape[-1] / float(sr)
        except Exception as e:
            print(f"无法读取音频文件 {audio_path}: {e}")
            return None

        # 处理所有语音片段，生成完整文本
        full_segments = []
        full_text = ""
        all_sentences = []

        for segment in conversation_data['result']['data']:
            try:
                text = segment['note']['text'].strip()
                cleaned_text = clean_text(text)

                if not cleaned_text:
                    continue

                start_time = float(segment['start'])
                end_time = float(segment['end'])
                duration = end_time - start_time

                if duration < args.min_duration:
                    continue

                if punctuation_model and cleaned_text:
                    try:
                        cleaned_text = punctuation_model(text_in=cleaned_text)['text']
                    except Exception as e:
                        print(f"标点符号恢复失败: {e}")

                full_text += cleaned_text
                all_sentences.append({
                    'start': start_time,
                    'end': end_time,
                    'text': cleaned_text
                })

            except Exception as e:
                print(f"处理语音片段时出错: {e}")
                continue

        if not full_text or total_duration < args.long_min_duration:
            return None

        # 创建完整音频的训练样本
        full_audio_data = {
            "audio": {
                "path": audio_path,
                "start_time": 0,
                "end_time": total_duration
            },
            "sentence": full_text,
            "duration": round(total_duration, 2),
            "sentences": all_sentences,
            "audio_filename": audio_filename,
            "data_type": "full_audio"
        }

        return [full_audio_data]

    except Exception as e:
        print(f"处理完整音频时出错: {e}")
        return None


def process_conversation(conversation_data, audio_base_dir, punctuation_model=None):
    """处理单个对话记录，生成各种类型的训练样本"""
    try:
        voice_url = conversation_data['data']['voice_url']
        audio_filename = voice_url.split('/')[-1]
        audio_path = os.path.join(audio_base_dir, audio_filename)

        if not os.path.exists(audio_path):
            print(f"警告: 音频文件不存在: {audio_path}")
            return None, None

        try:
            sample, sr = soundfile.read(audio_path)
            total_duration = sample.shape[-1] / float(sr)
        except Exception as e:
            print(f"无法读取音频文件 {audio_path}: {e}")
            return None, None

        # 处理短片段
        short_segments = []
        for segment in conversation_data['result']['data']:
            try:
                text = segment['note']['text'].strip()
                cleaned_text = clean_text(text)

                if not cleaned_text:
                    continue

                start_time = float(segment['start'])
                end_time = float(segment['end'])
                duration = end_time - start_time

                if duration < args.min_duration or duration > args.max_duration:
                    continue

                if punctuation_model and cleaned_text:
                    try:
                        cleaned_text = punctuation_model(text_in=cleaned_text)['text']
                    except Exception as e:
                        print(f"标点符号恢复失败: {e}")

                speaker = "unknown"
                role = "unknown"
                for attr in segment['note']['attr']:
                    if attr['header'] == 'speaker' and attr['value']:
                        speaker = attr['value']
                    elif attr['header'] == 'role' and attr['value']:
                        role = attr['value']

                segment_data = {
                    "audio": {
                        "path": audio_path,
                        "start_time": start_time,
                        "end_time": end_time
                    },
                    "sentence": cleaned_text,
                    "duration": round(duration, 2),
                    "sentences": [{
                        "start": 0,
                        "end": duration,
                        "text": cleaned_text
                    }],
                    "speaker": speaker,
                    "role": role,
                    "audio_filename": audio_filename,
                    "segment_id": segment.get('id', ''),
                    "data_type": "short_segment"
                }

                if duration <= args.short_max_duration:
                    short_segments.append(segment_data)

            except Exception as e:
                print(f"处理语音片段时出错: {e}")
                continue

        # 处理长片段（合并相邻片段）
        all_segments = []
        for segment in conversation_data['result']['data']:
            try:
                text = segment['note']['text'].strip()
                cleaned_text = clean_text(text)

                if not cleaned_text:
                    continue

                start_time = float(segment['start'])
                end_time = float(segment['end'])
                duration = end_time - start_time

                if duration < args.min_duration:
                    continue

                speaker = "unknown"
                role = "unknown"
                for attr in segment['note']['attr']:
                    if attr['header'] == 'speaker' and attr['value']:
                        speaker = attr['value']
                    elif attr['header'] == 'role' and attr['value']:
                        role = attr['value']

                segment_data = {
                    "audio": {"path": audio_path},
                    "sentence": cleaned_text,
                    "duration": round(duration, 2),
                    "sentences": [{
                        "start": start_time,
                        "end": end_time,
                        "text": cleaned_text
                    }],
                    "speaker": speaker,
                    "role": role,
                    "audio_filename": audio_filename,
                    "segment_id": segment.get('id', '')
                }

                all_segments.append(segment_data)

            except Exception as e:
                continue

        # 合并生成长片段
        long_segments = merge_segments(all_segments, max_duration=300.0, max_gap=args.segment_merge_gap)

        # 为长片段添加数据类型标记
        for seg in long_segments:
            seg['data_type'] = 'long_segment'

        return short_segments, long_segments

    except Exception as e:
        print(f"处理对话数据时出错: {e}")
        return None, None


def create_mixed_dataset(json_file, audio_dir, target_dir, num_workers=None):
    """创建混合数据集：完整音频 + 长片段 + 短片段 + 数据增强"""
    print('Create 95598 mixed dataset with full audio, segments and augmentation...')

    punctuation_model = load_punctuation_model()

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print(f"读取JSON文件: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    print(f"总共 {len(conversations)} 个对话记录")

    # 处理数据
    all_short_segments = []
    all_long_segments = []
    all_full_audio = []

    print("处理数据...")
    for conversation in tqdm(conversations, desc="处理对话"):
        # 处理短片段和长片段
        short_segments, long_segments = process_conversation(conversation, audio_dir, punctuation_model)

        if short_segments:
            all_short_segments.extend(short_segments)
        if long_segments:
            all_long_segments.extend(long_segments)

        # 处理完整音频
        if args.use_full_audio:
            full_audio_segments = process_full_audio(conversation, audio_dir, punctuation_model)
            if full_audio_segments:
                all_full_audio.extend(full_audio_segments)

    print(f"处理完成:")
    print(f"  短片段: {len(all_short_segments)} 个")
    print(f"  长片段: {len(all_long_segments)} 个")
    print(f"  完整音频: {len(all_full_audio)} 个")

    if not (all_short_segments or all_long_segments or all_full_audio):
        print("没有找到有效的数据")
        return

    # 按音频文件分组，确保训练集和测试集不重叠
    def split_by_audio_files(segments):
        if not segments:
            return [], [], set(), set()

        conversation_ids = list(set(seg['audio_filename'] for seg in segments))
        random.shuffle(conversation_ids)  # 随机打乱
        split_point = int(len(conversation_ids) * 0.8)

        train_conversations = set(conversation_ids[:split_point])
        test_conversations = set(conversation_ids[split_point:])

        train_segments = [seg for seg in segments if seg['audio_filename'] in train_conversations]
        test_segments = [seg for seg in segments if seg['audio_filename'] in test_conversations]

        return train_segments, test_segments, train_conversations, test_conversations

    # 分别划分数据集
    short_train, short_test, short_train_conv, short_test_conv = split_by_audio_files(all_short_segments)
    long_train, long_test, long_train_conv, long_test_conv = split_by_audio_files(all_long_segments)
    full_train, full_test, full_train_conv, full_test_conv = split_by_audio_files(all_full_audio)

    print(f"\n数据集划分:")
    print(f"短片段 - 训练: {len(short_train)}, 测试: {len(short_test)}")
    print(f"长片段 - 训练: {len(long_train)}, 测试: {len(long_test)}")
    print(f"完整音频 - 训练: {len(full_train)}, 测试: {len(full_test)}")

    # 创建混合训练集
    mixed_train = []
    mixed_test = []

    # 按比例混合：完整音频(40%) + 长片段(35%) + 短片段(25%)
    mixed_train.extend(full_train)
    mixed_train.extend(long_train)
    mixed_train.extend(short_train)

    # 测试集：使用所有类型的测试数据
    mixed_test.extend(full_test)
    mixed_test.extend(long_test)
    mixed_test.extend(short_test)

    print(f"\n混合数据集:")
    print(f"训练集: {len(mixed_train)} 个样本")
    print(f"测试集: {len(mixed_test)} 个样本")

    # 数据增强
    if args.data_augmentation:
        print(f"\n开始数据增强（倍数: {args.augmentation_ratio}）...")
        augmented_train = []

        for segment in tqdm(mixed_train, desc="数据增强"):
            # 只对短片段和中等长度的长片段进行增强，避免完整音频过度增强
            if segment['data_type'] in ['short_segment', 'long_segment'] and segment['duration'] < 60:
                try:
                    # 读取音频进行增强
                    audio_path = segment['audio']['path']

                    # 提取音频片段
                    if 'start_time' in segment['audio']:
                        start_time = segment['audio']['start_time']
                        end_time = segment['audio']['end_time']
                        audio, sr = librosa.load(audio_path, sr=16000,
                                                offset=start_time,
                                                duration=end_time-start_time)
                    else:
                        audio, sr = librosa.load(audio_path, sr=16000)

                    # 生成增强样本
                    augmented_audios = augment_audio(audio, sr)

                    # 创建增强样本
                    for i, aug_audio in enumerate(augmented_audios[1:], 1):  # 跳过原始音频
                        aug_segment = segment.copy()
                        aug_segment['audio_filename'] = f"{segment['audio_filename']}_aug_{i}"
                        aug_segment['segment_id'] = f"{segment.get('segment_id', '')}_aug_{i}"
                        aug_segment['data_type'] = f"{segment['data_type']}_augmented"

                        augmented_train.append(aug_segment)

                        # 控制增强数量
                        if len(augmented_train) >= len(mixed_train) * (args.augmentation_ratio - 1):
                            break

                except Exception as e:
                    print(f"增强音频时出错: {e}")
                    continue

        mixed_train.extend(augmented_train)
        print(f"数据增强完成，增加 {len(augmented_train)} 个样本")
        print(f"最终训练集: {len(mixed_train)} 个样本")

    # 写入数据集文件
    def write_json_file(filename, data):
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 写入各种数据集
    write_json_file(os.path.join(target_dir, 'mixed_train.json'), mixed_train)
    write_json_file(os.path.join(target_dir, 'mixed_test.json'), mixed_test)

    # 也保存分开的数据集，方便对比实验
    write_json_file(os.path.join(target_dir, 'short_train.json'), short_train)
    write_json_file(os.path.join(target_dir, 'short_test.json'), short_test)
    write_json_file(os.path.join(target_dir, 'long_train.json'), long_train)
    write_json_file(os.path.join(target_dir, 'long_test.json'), long_test)
    write_json_file(os.path.join(target_dir, 'full_train.json'), full_train)
    write_json_file(os.path.join(target_dir, 'full_test.json'), full_test)

    # 生成统计报告
    stats_file = os.path.join(target_dir, 'dataset_stats_v3.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("95598智能客服语音数据集统计报告 (V3 - 混合策略)\n")
        f.write("=" * 60 + "\n")
        f.write(f"总对话数: {len(conversations)}\n")
        f.write(f"短片段数: {len(all_short_segments)}\n")
        f.write(f"长片段数: {len(all_long_segments)}\n")
        f.write(f"完整音频数: {len(all_full_audio)}\n\n")

        f.write("训练集统计:\n")
        f.write(f"  短片段: {len(short_train)} 个\n")
        f.write(f"  长片段: {len(long_train)} 个\n")
        f.write(f"  完整音频: {len(full_train)} 个\n")
        f.write(f"  混合训练集: {len(mixed_train)} 个\n\n")

        f.write("测试集统计:\n")
        f.write(f"  短片段: {len(short_test)} 个\n")
        f.write(f"  长片段: {len(long_test)} 个\n")
        f.write(f"  完整音频: {len(full_test)} 个\n")
        f.write(f"  混合测试集: {len(mixed_test)} 个\n")

        # 时长统计
        def get_duration_stats(segments, name):
            if not segments:
                return
            durations = [seg['duration'] for seg in segments]
            f.write(f"\n{name}时长统计:\n")
            f.write(f"  样本数: {len(durations)}\n")
            f.write(f"  最短: {min(durations):.2f}秒\n")
            f.write(f"  最长: {max(durations):.2f}秒\n")
            f.write(f"  平均: {sum(durations)/len(durations):.2f}秒\n")

        get_duration_stats(mixed_train, "混合训练集")
        get_duration_stats(mixed_test, "混合测试集")
        get_duration_stats(all_full_audio, "完整音频")

    print("\n数据集创建完成！")
    print(f"混合训练集: {os.path.join(target_dir, 'mixed_train.json')}")
    print(f"混合测试集: {os.path.join(target_dir, 'mixed_test.json')}")
    print(f"统计报告: {stats_file}")


def main():
    print_arguments(args)

    num_workers = args.num_workers if args.num_workers else cpu_count()
    print(f"将使用 {num_workers} 个CPU核心进行并行处理")

    if not os.path.exists(args.json_file):
        print(f"错误: JSON文件不存在: {args.json_file}")
        return

    if not os.path.exists(args.audio_dir):
        print(f"错误: 音频目录不存在: {args.audio_dir}")
        return

    create_mixed_dataset(
        json_file=args.json_file,
        audio_dir=args.audio_dir,
        target_dir=args.target_dir,
        num_workers=num_workers
    )


if __name__ == '__main__':
    main()