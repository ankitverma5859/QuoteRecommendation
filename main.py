import language_models as lm
import nltk
import math
#nltk.download()

if __name__ == '__main__':

    #lm_uq = lm.UnigramModel('Books/PrideAndPrejudice.txt')
    #lm_uq.preprocess()

    lm_uq = lm.UnigramModel('Quotations/Quotations.txt')
    lm_uq.preprocess()
    lm_uq.fit()

    lm_uc = lm.UnigramModel('Books/PrideAndPrejudice.txt')
    lm_uc.preprocess()
    lm_uc.fit()

    print(f'Q:{lm_uq.get_probability("end")} C:{lm_uc.get_probability("end")} R:{math.log(lm_uq.get_probability("end") / lm_uc.get_probability("end"))}')
    print(f'Q:{lm_uq.get_probability("ability")} C:{lm_uc.get_probability("ability")} R:{math.log(lm_uq.get_probability("ability") / lm_uc.get_probability("ability"))}')
    print(f'Q:{lm_uq.get_probability("dello")} C:{lm_uc.get_probability("dello")} R:{math.log(lm_uq.get_probability("dello") / lm_uc.get_probability("dello"))}')
    print(f'Q:{lm_uq.get_probability("mello")} C:{lm_uc.get_probability("mello")} R:{math.log(lm_uq.get_probability("mello") / lm_uc.get_probability("mello"))}')





