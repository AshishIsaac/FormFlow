# Samples

Short openly licensed exercise clips from Wikimedia Commons (US Army training videos in the public domain, and CC BY / CC0 clips), each with FormFlow's annotated analysis (`*_annotated.mp4`). The clips were trimmed, scaled to 720p, re-encoded and stripped of audio; FormFlow's overlays were added to the annotated copies. Rebuild with `python scripts/fetch_samples.py DIR && python scripts/make_samples.py DIR`.

| file | content | exercise | FormFlow result | licence | author | source |
|---|---|---|---|---|---|---|
| `bench_press.mp4` | Bench press | bench_press | 1 reps, score 89.3 | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0) | FitnessScape | [Commons](https://commons.wikimedia.org/wiki/File:Bench_press_-_exercise_demonstration_video.webm) |
| `deadlift_barbell.mp4` | Barbell deadlift (rear view) | deadlift | 2 reps, score 100.0 | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0) | FitnessScape | [Commons](https://commons.wikimedia.org/wiki/File:Deadlift_-_exercise_demonstration_video.webm) |
| `deadlift_trapbar.mp4` | Trap-bar deadlift (3 reps, crowded field) | deadlift | 3 reps, score 78.3 | [Public domain](https://commons.wikimedia.org/wiki/File:Army_Combat_Fitness_Test-_3_Repetition_Maximum_Deadlift_(MDL)_(Event_1).webm) | Army Combat Fitness Test | [Commons](https://commons.wikimedia.org/wiki/File:Army_Combat_Fitness_Test-_3_Repetition_Maximum_Deadlift_(MDL)_(Event_1).webm) |
| `lunge_forward.mp4` | Forward lunge with kettlebells | lunge | 2 reps, score 95.0 | [Public domain](https://commons.wikimedia.org/wiki/File:Strength_Training_Circuit-_Forward_Lunge.webm) | Army Combat Fitness Test | [Commons](https://commons.wikimedia.org/wiki/File:Strength_Training_Circuit-_Forward_Lunge.webm) |
| `pullup.mp4` | Pull-ups (rear view) | pull_up | 1 reps, score 82.3 | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0) | FitnessScape | [Commons](https://commons.wikimedia.org/wiki/File:Pull-ups_-_exercise_demonstration_video.webm) |
| `push_press.mp4` | Kettlebell push press | overhead_press | 2 reps, score 51.5 | [Public domain](https://commons.wikimedia.org/wiki/File:Strength_Training_Circuit-_Overhead_Push_Press.webm) | Army Combat Fitness Test | [Commons](https://commons.wikimedia.org/wiki/File:Strength_Training_Circuit-_Overhead_Push_Press.webm) |
| `pushup_side.mp4` | Hand-release push-ups (side view, bystanders) | push_up | 2 reps, score 91.9 | [Public domain](https://commons.wikimedia.org/wiki/File:Army_Combat_Fitness_Test-_Hand-Release_Push-Up_(HRP)_(Event_3).webm) | Army Combat Fitness Test | [Commons](https://commons.wikimedia.org/wiki/File:Army_Combat_Fitness_Test-_Hand-Release_Push-Up_(HRP)_(Event_3).webm) |
| `shoulder_press.mp4` | Barbell shoulder press (rear view) | overhead_press | 2 reps, score 85.0 | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0) | FitnessScape | [Commons](https://commons.wikimedia.org/wiki/File:Shoulder_press_-_exercise_demonstration_video.webm) |
| `squat_barbell.mp4` | Barbell back squat | squat | 2 reps, score 100.0 | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0) | FitnessScape | [Commons](https://commons.wikimedia.org/wiki/File:Squat_-_exercise_demonstration_video.webm) |

`demo_synthetic.mp4` is rendered by `formflow demo` (synthetic athlete, no camera).
