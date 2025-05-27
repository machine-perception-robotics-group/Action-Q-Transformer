Action Q-Transformer
=======
Writer: Hidenori Itaya

This is the code for [Visual Explanation with Action Query Transformer in Deep Reinforcement Learning and Visual Feedback via Augmented Reality](aaaa).

※ This code is based on the following code: [Kaixhin/Rainbow](https://github.com/Kaixhin/Rainbow).


##  Requirements

1. Please obtain the atari2600 ROM.
    - Available Atari games can be found in the [atari-py ROMs folder](https://github.com/openai/atari-py/tree/master/atari_py/atari_roms) (Deprecated).

2. Please refer to the `requirements.txt` file or use the dockerhub image to build your environment.

    docker image of this code: 
    ```
    docker pull ita774/action-q-transformer
    ```


## Directory configuration

```
aqt_code/
├── main.py                 ：Train code for Action Q-Transformer.
├── main_ttq.py             ：Train code for Action Q-Transformer + Target Trained Q-Network.
├── eval.py                 ：Evaluate trained agent models (game scores).
├── eval_random.py          ：Evaluation of game score by random action selection.
├── eval_att.py             ：Code of attention visualization.
└── make_movie.py           ：Code to convert attention map or raw image to video.
```


## Training

Train code for agent model in Atari2600.

- Baseline model (Rainbow):
    ```
    python main.py --id **_rainbow --game ** --architecture canonical --T-max 50000000 --evaluation-interval 100000 --save-interval 500000 --cuda-device cuda:0 --memory
    ```

- Action Q-Transformer:
    ```
    python main.py --id **_aqt --game ** --architecture aqt --T-max 50000000 --evaluation-interval 100000 --save-interval 500000 --cuda-device cuda:0 --memory
    ```

- Action Q-Transformer with Target Trained Q-Netowrk:
  ```
  python main_ttq.py --id **_aqt_ttq --game ** --architecture aqt --T-max 50000000 --evaluation-interval 100000 --save-interval 500000 --ttq-target-model results/**_rainbow/models/best_model.pth --alpha-decay 25000000 --final-alpha 0.0 --cuda-device cuda:0 --memory
  ```
** is game task name. The `results` folder is created and the model is saved directly under it.


## Evaluation

- Eval code for agent model in Atari2600:
    ```
    python eval.py --game ** --architecture aqt --evaluation-episodes 100 --load-model model_path --cuda-device 0 
    ```

- Evaluation of scores in random action selection:
    ```
    python eval_random.py --evaluation-episodes 100 --game **
    ```
** is game task name.


## Visualization

Visualization code for attention or saliency of agent model in Atari2600. 

- Visualization attention:
    ```
    python eval_att.py --id save_folder_name --game *** --architecture aqt --evaluation-episodes 1 --load-model model_path --cuda-device cuda:0
    ```
    The `visuals` folder is created and the attention is saved directly under it.

- Visualization saliency:
  ```
  python eval_saliency.py --id save_folder_name --game ** --architecture aqt --evaluation-episodes 1 --load-model model_path --cuda-device cuda:0
  ```
  The `visuals_sal` folder is created and the attention is saved directly under it.


## Trained models

| game task                                         | link                                      |
|---------------------------------------------------|-------------------------------------------|
| [Alien](https://www.gymlibrary.dev/environments/atari/alien/)     |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/alien.zip) |
| [Amidar](https://www.gymlibrary.dev/environments/atari/amidar/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/amidar.zip) |
| [Assault](https://www.gymlibrary.dev/environments/atari/assault/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/assault.zip) |
| [Asterix](https://www.gymlibrary.dev/environments/atari/asterix/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/asterix.zip) |
| [Asteroids](https://www.gymlibrary.dev/environments/atari/asteroids/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/asteroids.zip) |
| [Atlantis](https://www.gymlibrary.dev/environments/atari/atlantis/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/atlantis.zip) |
| [Bank Heist](https://www.gymlibrary.dev/environments/atari/bank_heist/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/bank_heist.zip) |
| [Battle Zone](https://www.gymlibrary.dev/environments/atari/battle_zone/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/battle_zone.zip) |
| [Beam Rider](https://www.gymlibrary.dev/environments/atari/beam_rider/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/beam_rider.zip) |
| [Berzerk](https://www.gymlibrary.dev/environments/atari/berzerk/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/berzerk.zip) |
| [Bowling](https://www.gymlibrary.dev/environments/atari/bowling/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/bowling.zip) |
| [Boxing](https://www.gymlibrary.dev/environments/atari/boxing/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/boxing.zip) |
| [Breakout](https://www.gymlibrary.dev/environments/atari/breakout/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/breakout.zip) |
| [Centipede](https://www.gymlibrary.dev/environments/atari/centipede/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/centipede.zip) |
| [Chopper Command](https://www.gymlibrary.dev/environments/atari/chopper_command/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/chopper_command.zip) |
| [Crazy Climber](https://www.gymlibrary.dev/environments/atari/crazy_climber/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/crazy_climber.zip) |
| [Defender](https://www.gymlibrary.dev/environments/atari/defender/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/defender.zip) |
| [Demon Attack](https://www.gymlibrary.dev/environments/atari/demon_attack/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/demon_attack.zip) |
| [Double Dunk](https://www.gymlibrary.dev/environments/atari/double_dunk/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/double_dunk.zip) |
| [Enduro](https://www.gymlibrary.dev/environments/atari/enduro/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/enduro.zip) |
| [FishingDerby](https://www.gymlibrary.dev/environments/atari/fishing_derby/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/fishing_derby.zip) |
| [Freeway](https://www.gymlibrary.dev/environments/atari/freeway/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/freeway.zip) |
| [Frostbite](https://www.gymlibrary.dev/environments/atari/frostbite/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/frostbite.zip) |
| [Gopher](https://www.gymlibrary.dev/environments/atari/gopher/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/gopher.zip) |
| [Gravitar](https://www.gymlibrary.dev/environments/atari/gravitar/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/gravitar.zip) |
| [Hero](https://www.gymlibrary.dev/environments/atari/hero/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/hero.zip) |
| [Kangaroo](https://www.gymlibrary.dev/environments/atari/kangaroo/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/kangaroo.zip) |
| [Krull](https://www.gymlibrary.dev/environments/atari/krull/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/krull.zip) |
| [Kung Fu Master](https://www.gymlibrary.dev/environments/atari/kung_fu_master/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/kung_fu_master.zip) |
| [Montezuma Revenge](https://www.gymlibrary.dev/environments/atari/montezuma_revenge/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/montezuma_revenge.zip) |
| [Ms Pacman](https://www.gymlibrary.dev/environments/atari/ms_pacman/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/ms_pacman.zip) |
| [Name This Game](https://www.gymlibrary.dev/environments/atari/name_this_game/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/name_this_game.zip) |
| [Phoenix](https://www.gymlibrary.dev/environments/atari/phoenix/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/phoenix.zip) |
| [Pitfall](https://www.gymlibrary.dev/environments/atari/pitfall/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/pitfall.zip) |
| [Pong](https://www.gymlibrary.dev/environments/atari/pong/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/pong.zip) |
| [Qbert](https://www.gymlibrary.dev/environments/atari/qbert/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/qbert.zip) |
| [Road Runner](https://www.gymlibrary.dev/environments/atari/road_runner/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/road_runner.zip) |
| [Robot Tank](https://www.gymlibrary.dev/environments/atari/robotank/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/robotank.zip) |
| [Seaquest](https://www.gymlibrary.dev/environments/atari/seaquest/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/seaquest.zip) |
| [Skiings](https://www.gymlibrary.dev/environments/atari/skiing/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/skiing.zip) |
| [Solaris](https://www.gymlibrary.dev/environments/atari/solaris/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/solaris.zip) |
| [SpaceInvaders](https://www.gymlibrary.dev/environments/atari/space_invaders/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/space_invaders.zip) |
| [StarGunner](https://www.gymlibrary.dev/environments/atari/star_gunner/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/star_gunner.zip) |
| [Tennis](https://www.gymlibrary.dev/environments/atari/tennis/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/tennis.zip) |
| [TimePilot](https://www.gymlibrary.dev/environments/atari/time_pilot/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/time_pilot.zip) |
| [Tutankham](https://www.gymlibrary.dev/environments/atari/tutankham/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/tutankham.zip) |
| [Venture](https://www.gymlibrary.dev/environments/atari/venture/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/venture.zip) |
| [Video Pinball](https://www.gymlibrary.dev/environments/atari/video_pinball/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/video_pinball.zip) |
| [Wizard of Wor](https://www.gymlibrary.dev/environments/atari/wizard_of_wor/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/wizard_of_wor.zip) |
| [Zaxxon](https://www.gymlibrary.dev/environments/atari/zaxxon/)   |  [download](http://www.mprg.cs.chubu.ac.jp/~itaya/share/trained_models/aqt/zaxxon.zip) |

