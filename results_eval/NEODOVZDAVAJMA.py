predict = open('nn_classification.txt', 'r')
predictions = predict.readlines()

real = open('targets.txt', 'r')
real_values = real.readlines()



total = 0
false_alm = 0
miss = 0

for prediction in predictions:
    total += 1
    values = prediction.split(' ')
    name, probability, prediction = values
    if name in real_values:
        if int(prediction) != 1:
            miss +=1
    else:
        if int(prediction) == 1:
            false_alm +=1
            print(name)

print("miss", miss, "false_alm",false_alm,"total %", 100 -  (false_alm+miss)/(total/100) )