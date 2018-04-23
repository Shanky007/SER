import numpy as np
from moviepy.audio.io.AudioFileClip import AudioFileClip
from pathlib import Path

root_dir = Path('RECOLA/')

portion_to_id = dict(
    train=[25, 16, 17, 21, 23, 37, 39, 41, 46, 56],
    valid=[19, 26, 28, 30, 34, 42, 43, 45, 64, 65],
    test=[43, 45, 64, 65, 48, 58, 62]
)


def main(directory):
  for portion in portion_to_id.keys():
    xs =[]
    ys = None
    for subj_id in portion_to_id[portion]:
        subject_name = 'P{}'.format(subj_id)
        audio, label = get_samples(subject_name)
        xs+=audio
        ys = np.concatenate((ys,label)) if not ys==None else label

        # for i, (audio, label) in enumerate(zip(*get_samples(subject_name))):
        #     ar.append([[i],[subj_id],audio, label])
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    print(portion)
    print(np.shape(xs))
    print(np.shape(ys))
    np.save(portion+"_input.npy", xs)
    np.save(portion+"_output.npy", ys)


def get_samples(subject_id):
    arousal_label_path = root_dir / 'Ratings_affective_behaviour_CCC_centred/arousal/{}.csv'.format(subject_id)
    valence_label_path = root_dir / 'Ratings_affective_behaviour_CCC_centred/valence/{}.csv'.format(subject_id)

    clip = AudioFileClip(str(root_dir / "Audio_recordings_WAV/{}.wav".format(subject_id)))

    subsampled_audio = clip.set_fps(16000)

    audio_frames = []
    for i in range(1, 7501):
        time = 0.04 * i

        audio = np.array(list(subsampled_audio.subclip(time - 0.04, time).iter_frames())) #count the number of audio frames
        audio = audio.mean(1)[:640]  # takes the mean along axis=1 (along the row)

        audio_frames.append(audio.astype(np.float32))

    arousal = np.loadtxt(str(arousal_label_path), delimiter=';', skiprows=1)[:, 1][1:]

    valence = np.loadtxt(str(valence_label_path), delimiter=';', skiprows=1)[:, 1][1:]

    return audio_frames, np.dstack([arousal, valence])[0].astype(np.float32)

if __name__ == "__main__":
    main(Path('records/'))
