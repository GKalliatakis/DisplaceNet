from __future__ import print_function
import numpy as np

def _obtain_emotional_traits_calibrated_predictions(emotional_trait,
                                                    raw_preds):



    # print('[INFO] Violation before valence & dominance: ',raw_preds[0][0] )
    # print('[INFO] No-violation before valence & dominance: ', raw_preds[0][1])

    #Dominance
    # neutral interval -- the same raw predictions will be returned
    if 4.5 <= emotional_trait <= 5.5:
        # print('neutral dominance')
        violation = raw_preds[0][0]
        no_violation = raw_preds[0][1]


    # positive dominance
    elif emotional_trait > 5.5:
        # print('positive dominance')
        diff_from_neutral = emotional_trait-5.5
        adjustment = diff_from_neutral * 0.11
        # adjustment = diff_from_neutral * 0.05
        violation = raw_preds[0][0]-adjustment
        no_violation = raw_preds[0][1]+adjustment

    # negative dominance
    elif emotional_trait < 4.5:
        # print ('negative dominance')
        diff_from_neutral = 4.5-emotional_trait
        adjustment = diff_from_neutral * 0.11
        # adjustment = diff_from_neutral * 0.05
        violation = raw_preds[0][0]+adjustment
        no_violation = raw_preds[0][1]-adjustment


    #
    # print('[INFO] Violation after dominance: ',violation )
    # print('[INFO] No-violation after dominance: ', no_violation)


    calibrated_preds = np.array([[violation, no_violation]])


    return calibrated_preds











