import argparse
import functools
import json
import os
import re
from multiprocessing import Pool, cpu_count

import soundfile
from tqdm import tqdm

from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("json_file", default="95598智能客服语音数据集/json_19343_20221009173201.json",
        type=str, help="95598数据集JSON标注文件路径")
add_arg("audio_dir", default="95598智能客服语音数据集/广州局95598/",
        type=str, help="音频文件所在目录")
add_arg("target_dir", default="dataset/95598/", type=str, help="输出标注文件的目录")
add_arg('add_pun', default=False, type=bool, help="是否添加标点符")
add_arg('num_workers', default=None, type=int, help="并行处理的进程数，默认使用CPU核心数")
add_arg('min_duration', default=0.5, type=float, help="过滤过短的语音片段（秒）")
add_arg('max_duration', default=30.0, type=float, help="过滤过长的语音片段（秒）")
add_arg('short_max_duration', default=60.0, type=float, help="短片段最大时长（秒）")
add_arg('long_min_duration', default=60.0, type=float, help="长片段最小时长（秒）")
add_arg('long_max_duration', default=3000.0, type=float, help="长片段最大时长（秒）")
add_arg('merge_gap', default=30.0, type=float, help="合并相邻片段的最大时间间隔（秒）")
add_arg('create_long_data', default=True, type=bool, help="是否创建长片段训练数据")
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
    # 移除方括号内的特殊标记，如[ENS], [NIT]等
    text = re.sub(r'\[.*?\]', '', text)
    # 移除多余的空格
    text = re.sub(r'\s+', '', text)
    return text.strip()


def merge_segments(segments, max_duration=30.0, max_gap=3.0):
    """合并相邻的语音片段，生成更长的训练样本"""
    if not segments:
        return []

    # 按开始时间排序
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

        # 检查是否可以合并
        can_merge = False

        if not current_merge['sentences']:
            # 第一个片段
            can_merge = True
        else:
            # 计算当前合并片段的结束时间和新片段的开始时间间隔
            current_end = current_merge['sentences'][-1]['end']
            gap = segment_start - current_end

            # 判断是否可以合并
            if (gap <= max_gap and
                current_merge['duration'] + segment_duration + gap <= max_duration):
                can_merge = True

        if can_merge:
            # 合并片段
            current_merge['sentences'].extend(segment['sentences'])
            current_merge['sentence'] += segment['sentence']
            current_merge['duration'] += segment_duration + (segment_start - current_merge['sentences'][0]['start'] if current_merge['sentences'] else 0)
            current_merge['segment_ids'].append(segment.get('segment_id', ''))

            # 更新时间戳（以第一个片段的开始时间为基准）
            if len(current_merge['sentences']) > 1:
                base_start = current_merge['sentences'][0]['start']
                for sent in current_merge['sentences']:
                    # 时间戳相对于合并后的片段开始时间
                    pass  # 保持原始时间戳
        else:
            # 不能合并，保存当前合并结果
            if current_merge['duration'] >= args.long_min_duration:
                # 重新计算合并后的准确时间戳
                if current_merge['sentences']:
                    merged_start = current_merge['sentences'][0]['start']
                    merged_end = current_merge['sentences'][-1]['end']
                    current_merge['duration'] = round(merged_end - merged_start, 2)

                    # 调整时间戳为相对于合并片段的偏移
                    for sent in current_merge['sentences']:
                        sent['start'] = round(sent['start'] - merged_start, 3)
                        sent['end'] = round(sent['end'] - merged_start, 3)

                merged_segments.append(current_merge)

            # 开始新的合并
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

    # 处理最后一个合并片段
    if current_merge['duration'] >= args.long_min_duration:
        if current_merge['sentences']:
            merged_start = current_merge['sentences'][0]['start']
            merged_end = current_merge['sentences'][-1]['end']
            current_merge['duration'] = round(merged_end - merged_start, 2)

            # 调整时间戳
            for sent in current_merge['sentences']:
                sent['start'] = round(sent['start'] - merged_start, 3)
                sent['end'] = round(sent['end'] - merged_start, 3)

        merged_segments.append(current_merge)

    return merged_segments


def process_conversation(conversation_data, audio_base_dir, punctuation_model=None):
    """处理单个对话记录"""
    try:
        # 提取音频文件路径
        voice_url = conversation_data['data']['voice_url']
        # 从URL中提取文件名，如 "hw202001160924057_23367.wav"
        audio_filename = voice_url.split('/')[-1]
        audio_path = os.path.join(audio_base_dir, audio_filename)

        # 检查音频文件是否存在
        if not os.path.exists(audio_path):
            print(f"警告: 音频文件不存在: {audio_path}")
            return None

        # 验证音频文件是否可读
        try:
            sample, sr = soundfile.read(audio_path)
            total_duration = sample.shape[-1] / float(sr)
        except Exception as e:
            print(f"无法读取音频文件 {audio_path}: {e}")
            return None

        # 处理该对话的所有语音片段
        processed_segments = []

        for segment in conversation_data['result']['data']:
            try:
                # 提取文本内容
                text = segment['note']['text'].strip()

                # 清理文本，移除特殊标记
                cleaned_text = clean_text(text)

                # 跳过空文本或纯特殊标记
                if not cleaned_text:
                    continue

                # 获取时间戳
                start_time = float(segment['start'])
                end_time = float(segment['end'])
                duration = end_time - start_time

                # 过滤过短或过长的片段
                if duration < args.min_duration or duration > args.max_duration:
                    continue

                # 可选：添加标点符号
                if punctuation_model and cleaned_text:
                    try:
                        cleaned_text = punctuation_model(text_in=cleaned_text)['text']
                    except Exception as e:
                        print(f"标点符号恢复失败: {e}")

                # 提取说话人信息
                speaker = "unknown"
                role = "unknown"

                for attr in segment['note']['attr']:
                    if attr['header'] == 'speaker' and attr['value']:
                        speaker = attr['value']
                    elif attr['header'] == 'role' and attr['value']:
                        role = attr['value']

                # 构造与AISHELL兼容的数据格式
                segment_data = {
                    "audio": {
                        "path": audio_path
                    },
                    "sentence": cleaned_text,
                    "duration": round(duration, 2),
                    "sentences": [{
                        "start": start_time,
                        "end": end_time,
                        "text": cleaned_text
                    }],
                    # 额外的元数据
                    "speaker": speaker,
                    "role": role,
                    "audio_filename": audio_filename,
                    "segment_id": segment.get('id', ''),
                    "conversation_audio_duration": total_duration
                }

                processed_segments.append(segment_data)

            except Exception as e:
                print(f"处理语音片段时出错: {e}")
                continue

        return processed_segments

    except Exception as e:
        print(f"处理对话数据时出错: {e}")
        return None


def create_annotation_text(json_file, audio_dir, target_dir, num_workers=None):
    """创建95598数据集的标注文件，包括短片段和长片段"""
    print('Create 95598 annotation text with short and long segments...')

    # 加载标点符号恢复模型（如果需要）
    punctuation_model = load_punctuation_model()

    # 创建输出目录
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 读取JSON数据
    print(f"读取JSON文件: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    print(f"总共 {len(conversations)} 个对话记录")

    # 处理所有对话的语音片段
    all_segments = []

    print("处理语音数据...")
    for conversation in tqdm(conversations, desc="处理对话"):
        segments = process_conversation(conversation, audio_dir, punctuation_model)
        if segments:
            all_segments.extend(segments)

    print(f"总共处理了 {len(all_segments)} 个有效语音片段")

    if not all_segments:
        print("没有找到有效的语音片段，请检查数据格式")
        return

    # 按音频文件分组
    segments_by_file = {}
    for segment in all_segments:
        filename = segment['audio_filename']
        if filename not in segments_by_file:
            segments_by_file[filename] = []
        segments_by_file[filename].append(segment)

    # 分离短片段和生成长片段
    short_segments = []
    long_segments = []

    print("生成短片段数据集...")
    for filename, segments in segments_by_file.items():
        for segment in segments:
            if segment['duration'] <= args.short_max_duration:
                short_segments.append(segment)

    print(f"生成 {len(short_segments)} 个短片段 (<= {args.short_max_duration}秒)")

    if args.create_long_data:
        print("生成长片段数据集...")
        for filename, segments in segments_by_file.items():
            # 过滤出可以合并的片段
            mergeable_segments = [seg for seg in segments if seg['duration'] > 0]
            if mergeable_segments:
                # 合并相邻片段
                merged = merge_segments(mergeable_segments,
                                      max_duration=args.long_max_duration,
                                      max_gap=args.merge_gap)
                long_segments.extend(merged)

        print(f"生成 {len(long_segments)} 个长片段 ({args.long_min_duration}-{args.long_max_duration}秒)")

    # 按照对话ID进行训练集和测试集划分
    def split_segments(segments):
        """分割训练集和测试集"""
        conversation_ids = list(set(seg['audio_filename'] for seg in segments))
        split_point = int(len(conversation_ids) * 0.8)

        train_conversations = set(conversation_ids[:split_point])
        test_conversations = set(conversation_ids[split_point:])

        train_segments = [seg for seg in segments if seg['audio_filename'] in train_conversations]
        test_segments = [seg for seg in segments if seg['audio_filename'] in test_conversations]

        return train_segments, test_segments, train_conversations, test_conversations

    # 分离短片段
    short_train, short_test, short_train_conv, short_test_conv = split_segments(short_segments)
    print(f"短片段训练集: {len(short_train)} 个片段, 来自 {len(short_train_conv)} 个对话")
    print(f"短片段测试集: {len(short_test)} 个片段, 来自 {len(short_test_conv)} 个对话")

    # 分离长片段
    if args.create_long_data:
        long_train, long_test, long_train_conv, long_test_conv = split_segments(long_segments)
        print(f"长片段训练集: {len(long_train)} 个片段, 来自 {len(long_train_conv)} 个对话")
        print(f"长片段测试集: {len(long_test)} 个片段, 来自 {len(long_test_conv)} 个对话")

    # 写入短片段数据集
    short_train_file = os.path.join(target_dir, 'short_train.json')
    short_test_file = os.path.join(target_dir, 'short_test.json')

    with open(short_train_file, 'w', encoding='utf-8') as f:
        for segment in short_train:
            f.write(json.dumps(segment, ensure_ascii=False) + '\n')

    with open(short_test_file, 'w', encoding='utf-8') as f:
        for segment in short_test:
            f.write(json.dumps(segment, ensure_ascii=False) + '\n')

    # 写入长片段数据集
    if args.create_long_data:
        long_train_file = os.path.join(target_dir, 'long_train.json')
        long_test_file = os.path.join(target_dir, 'long_test.json')

        with open(long_train_file, 'w', encoding='utf-8') as f:
            for segment in long_train:
                f.write(json.dumps(segment, ensure_ascii=False) + '\n')

        with open(long_test_file, 'w', encoding='utf-8') as f:
            for segment in long_test:
                f.write(json.dumps(segment, ensure_ascii=False) + '\n')

    # 生成混合训练集（短片段+长片段）
    mixed_train_file = os.path.join(target_dir, 'mixed_train.json')
    with open(mixed_train_file, 'w', encoding='utf-8') as f:
        # 写入短片段
        for segment in short_train:
            f.write(json.dumps(segment, ensure_ascii=False) + '\n')
        # 写入长片段
        if args.create_long_data:
            for segment in long_train:
                f.write(json.dumps(segment, ensure_ascii=False) + '\n')

    print(f"混合训练集: {len(short_train) + (len(long_train) if args.create_long_data else 0)} 个片段")

    # 生成数据统计报告
    stats_file = os.path.join(target_dir, 'mixed_dataset_stats.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("95598智能客服语音数据集统计报告（混合长短片段）\n")
        f.write("=" * 50 + "\n")
        f.write(f"总对话数: {len(conversations)}\n")
        f.write(f"总语音片段数: {len(all_segments)}\n\n")

        # 短片段统计
        f.write("短片段统计:\n")
        f.write(f"  训练集: {len(short_train)} 个片段, 来自 {len(short_train_conv)} 个对话\n")
        f.write(f"  测试集: {len(short_test)} 个片段, 来自 {len(short_test_conv)} 个对话\n")

        # 长片段统计
        if args.create_long_data:
            f.write("\n长片段统计:\n")
            f.write(f"  训练集: {len(long_train)} 个片段, 来自 {len(long_train_conv)} 个对话\n")
            f.write(f"  测试集: {len(long_test)} 个片段, 来自 {len(long_test_conv)} 个对话\n")
            f.write(f"\n混合训练集: {len(short_train) + len(long_train)} 个片段\n")

        # 时长统计
        all_durations = [seg['duration'] for seg in all_segments]
        short_durations = [seg['duration'] for seg in short_segments]

        f.write(f"\n时长统计:\n")
        f.write(f"  全部片段: {len(all_durations)} 个, 平均时长: {sum(all_durations)/len(all_durations):.2f}秒\n")
        f.write(f"  短片段: {len(short_durations)} 个, 平均时长: {sum(short_durations)/len(short_durations):.2f}秒\n")

        if args.create_long_data:
            long_durations = [seg['duration'] for seg in long_segments]
            f.write(f"  长片段: {len(long_durations)} 个, 平均时长: {sum(long_durations)/len(long_durations):.2f}秒\n")

    print("混合数据集创建完成！")
    print(f"短片段训练集: {short_train_file}")
    print(f"短片段测试集: {short_test_file}")
    if args.create_long_data:
        print(f"长片段训练集: {long_train_file}")
        print(f"长片段测试集: {long_test_file}")
    print(f"混合训练集: {mixed_train_file}")
    print(f"统计报告: {stats_file}")


def main():
    print_arguments(args)

    num_workers = args.num_workers if args.num_workers else cpu_count()
    print(f"将使用 {num_workers} 个CPU核心进行并行处理")

    # 验证输入文件
    if not os.path.exists(args.json_file):
        print(f"错误: JSON文件不存在: {args.json_file}")
        return

    if not os.path.exists(args.audio_dir):
        print(f"错误: 音频目录不存在: {args.audio_dir}")
        return

    create_annotation_text(
        json_file=args.json_file,
        audio_dir=args.audio_dir,
        target_dir=args.target_dir,
        num_workers=num_workers
    )


if __name__ == '__main__':
    main()