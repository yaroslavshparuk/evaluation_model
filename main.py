import word
import words_model
import numpy as np
import lwa

# words model for colleague evaluation words
model = words_model.colleague_evaluation_model

# mock feedback from other colleagues
other_colleagues_feedback = ['Very good', 'Good', 'Fair', 'Bad', 'Very good']

# Calc
W = [other_colleagues_feedback.count(item) for item in model['words']]

# define minimal value for LMF
h = min(fou['lmf'][-1] for fou in model['words'].values())
m = 50

# Calc alpha cuts for UMF and LMF
intervals_umf = lwa.alpha_cuts_intervals(m)
intervals_lmf = lwa.alpha_cuts_intervals(m, h)

# calc Y values for UMF and LMF
res_lmf = lwa.y_lmf(intervals_lmf, model, W)
res_umf = lwa.y_umf(intervals_umf, model, W)

# visualization
res = lwa.construct_dit2fs(np.arange(*model['x']), intervals_lmf, res_lmf, intervals_umf, res_umf)
res.plot()

# calc the similarity measure 
sm = []
for title, fou in model['words'].items():
    word_obj = word.Word(None, model['x'], fou['lmf'], fou['umf'])
    similarity = res.similarity_measure(word_obj)
    sm.append((title, similarity))

# calc of the highest similarity feedback
res_word = max(sm, key=lambda item: item[1])
print(f"Most appropriate assessment is {res_word[0]} with similarity {res_word[1]:.2f}")