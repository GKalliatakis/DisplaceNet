# -*- coding: utf-8 -*-
'''
Use three emotional dimensions - pleasure, arousal and dominance - to describe human perceptions of physical environments.

Interpretations of pleasure: Positive versus negative affective states (e.g. excitement, relaxation, love, and
tranquility versus cruelty, humiliation, disinterest, and boredom)

Interpretations of arousal: Level of mental alertness and physical activity. (e.g. sleep, inactivity, boredom, and
relaxation at the lower end versus wakefulness, bodily tension, strenuous
exercise, and concentration at the higher end).

Interpretations of dominance: Ranges from feelings of total lack control or influence on events and surroundings to
the opposite extreme of feeling influential and in control

'''

from engine.human_centric_branch.global_emotional_traits_branch import single_img_VAD_inference, single_img_VAD_inference_with_bounding_boxes

img_path = '/home/sandbox/Desktop/canggu-honeymoon-photography-013.jpg'
model_a_backend_name = 'VGG16'
model_b_backend_name = None
model_c_backend_name = None

valence, arousal, dominance = single_img_VAD_inference(img_path=img_path,
                                                       object_detector_backend='RetinaNet',
                                                       model_a_backend_name=model_a_backend_name,
                                                       model_b_backend_name=model_b_backend_name,
                                                       model_c_backend_name=model_c_backend_name,
                                                       )

