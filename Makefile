

create_environment:
	conda env create -f environment.yml


download_dataset_CNN:
	rm -rf dataset
	mkdir -p dataset/train/
	mkdir -p dataset/dev/
	mkdir -p dataset/dev/data_img/non_target
	mkdir -p  dataset/dev/data_img/target
	mkdir -p  dataset/dev/data_sound/non_target
	mkdir -p  dataset/dev/data_sound/target
	mkdir -p dataset/train/data_img/non_target
	mkdir -p  dataset/train/data_img/target
	mkdir -p  dataset/train/data_sound/non_target
	mkdir -p  dataset/train/data_sound/target
	curl https://www.fit.vutbr.cz/study/courses/SUR/public/projekt_2021-2022/SUR_projekt2021-2022.zip -o dataset/SUR_projekt2021-2022.zip
	cd  dataset  && unzip SUR_projekt2021-2022.zip
	mv dataset/non_target_dev/*.png  dataset/dev/data_img/non_target/
	mv dataset/target_dev/*.png	dataset/dev/data_img/target/
	mv dataset/non_target_dev/*.wav dataset/dev/data_sound/non_target
	mv dataset/target_dev/*.wav dataset/dev/data_sound/target
	mv dataset/non_target_train/*.png  dataset/train/data_img/non_target/
	mv dataset/target_train/*.png	dataset/train/data_img/target/
	mv dataset/non_target_train/*.wav dataset/train/data_sound/non_target
	mv dataset/target_train/*.wav dataset/train/data_sound/target
	rm -rf dataset/non_target_train/
	rm -rf dataset/target_train/
	rm -rf dataset/non_target_dev/
	rm -rf dataset/target_dev/


train_nn:
	cd src/NN_img_classification/ && python3 train.py

make_eval:
	cd src/NN_img_classification/ && python3 eval_final_data_NN_classifications.py

make_graphs:
	cd src/NN_img_classification/ && python3 graphs.py
