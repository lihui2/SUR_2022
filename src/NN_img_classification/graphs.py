import matplotlib.pyplot as plt
import pickle

with open('training_data/training_data_with_transform.pickle', 'rb') as f:
    training_data_t = pickle.load(f)

with open('training_data/training_data_no_transform.pickle', 'rb') as f:
    training_data_nt = pickle.load(f)

num_of_correct_array = training_data_t["num_of_correct_array"]
num_of_samples_array = training_data_t["num_of_samples_array"]
false_alarms = training_data_t["false_alarams"]
misses = training_data_t["misses"]
epochs = training_data_t["epochs"]

fig, axs = plt.subplots(2, 1)
correctness = axs[0].plot(epochs,
                          [num_of_correct_array[x] / num_of_samples_array[0] for x in range(len(num_of_correct_array))])
axs[0].legend(['num_of_correct/num_of_samples'], shadow=True, fancybox=True)

# axs[0].set_xlim(0, 2)
axs[0].set_xlabel('epochs')
axs[0].set_ylabel('Correct %')

false_alarams_l, misses_l = axs[1].plot(epochs, false_alarms, epochs, misses)
axs[1].set_ylabel('Count')
axs[1].set_xlabel('epochs')
axs[1].legend((false_alarams_l, misses_l), ('false_alarms', 'misses'), shadow=True, fancybox=True)
fig.tight_layout()
plt.savefig('training1.png')
plt.show()

num_of_correct_array = training_data_nt["num_of_correct_array"]
num_of_samples_array = training_data_nt["num_of_samples_array"]
false_alarms = training_data_nt["false_alarams"]
misses = training_data_nt["misses"]
epochs = training_data_nt["epochs"]


fig, axs = plt.subplots(2, 1)
correctness = axs[0].plot(epochs,
                          [num_of_correct_array[x] / num_of_samples_array[0] for x in range(len(num_of_correct_array))])
axs[0].legend(['num_of_correct/num_of_samples'], shadow=True, fancybox=True)

# axs[0].set_xlim(0, 2)
axs[0].set_xlabel('epochs')
axs[0].set_ylabel('Correct %')

false_alarams_l, misses_l = axs[1].plot(epochs, false_alarms, epochs, misses)
axs[1].set_ylabel('Count')
axs[1].set_xlabel('epochs')
axs[1].legend((false_alarams_l, misses_l), ('false_alarms', 'misses'), shadow=True, fancybox=True)
fig.tight_layout()
plt.savefig('training2.png')
plt.show()

num_of_correct_array_t = training_data_t["num_of_correct_array"]
num_of_correct_array_nt = training_data_nt["num_of_correct_array"]

fig, axs = plt.subplots()
transforms, no_transforms = axs.plot(epochs,
         [num_of_correct_array_t[x] / num_of_samples_array[0] for x in range(len(num_of_correct_array_t))],
         epochs,
         [num_of_correct_array_nt[x] / num_of_samples_array[0] for x in range(len(num_of_correct_array_nt))]
         )

axs.legend((transforms, no_transforms), ('transforms', 'no_transforms'), shadow=True, fancybox=True)
axs.set_ylabel('Correct %')
axs.set_xlabel('Epochs')
plt.savefig('training3.png')
plt.show()