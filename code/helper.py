import numpy as np
import pandas as pd

def transform_score(score: np.ndarray):
    """
    对评分进行划分，划分为1-5
    :param score:
    :return:
    """
    data = pd.DataFrame({'score': score})
    scores1 = score.copy()
    scores1=np.sort(scores1)
    num_item = len(scores1)
    bin = num_item // 5

    def cut(val):
        if val <= scores1[bin]:
            return 1
        elif val <= scores1[bin * 2]:
            return 2
        elif val <= scores1[bin * 3]:
            return 3
        elif val <= scores1[bin * 4]:
            return 4
        return 5

    data['new_score'] = data['score'].apply(cut)
    return data['new_score']

if __name__ == '__main__':
    scores=np.random.randint(0,100,(200,))
    print(transform_score(scores))