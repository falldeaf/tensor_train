require('dotenv').config()
const os = require('os');
const path = require('path');
const fs = require('fs');
const jpeg = require('jpeg-js');
const tf = require('@tensorflow/tfjs');
const mobilenetModule = require('@tensorflow-models/mobilenet');
const knnClassifier = require('@tensorflow-models/knn-classifier');

let Client = require('ssh2-sftp-client');

let classifier, mobilenet;
const remote_path = '/training/plant';

const init = async function() {
	classifier = knnClassifier.create();
	mobilenet = await mobilenetModule.load();
}

const learn = async function() {

	await init();

	let client = new Client();

	const config = {
		host: process.env.host,
		port: 22,
		username: process.env.user,
		password: process.env.pass
	};
	
	await client.connect(config);

	let files = await client.list(remote_path);

	for (const file of files) {
		if(!file.name.startsWith("classifier") && file.name.endsWith(".json")) {
			let file_basename = path.basename(file.name, ".json");
			const image_path = remote_path + '/' + file_basename + '.jpg';
			console.log(image_path);

			//Get the classification label for this image from the metadata json file
			const cjson = JSON.parse(await client.get(remote_path + '/' + file.name));

			//Add the image training data
			await addImage(await client.get(image_path), cjson.classifier);

			console.log("image added");
		}
	}

	await client.end();
	console.log('conneciton closed');

	/*
	
	const classifier = knnClassifier.create();
	const mobilenet = await mobilenetModule.load();

	await addImage(fs.readFileSync('images/plant/plant_false (2).jpg'), "fine");
	await addImage(fs.readFileSync('images/plant/plant_false (3).jpg'), "fine");
	await addImage(fs.readFileSync('images/plant/plant_false (4).jpg'), "fine");
	await addImage(fs.readFileSync('images/plant/plant_true (3).jpg'), "dry");
	await addImage(fs.readFileSync('images/plant/plant_true (4).jpg'), "dry");
	await addImage(fs.readFileSync('images/plant/plant_true.jpg'), "dry");

	//await testImage(fs.readFileSync('images/plant/test_false.jpg'));

	console.log(await classifier.getClassifierDataset());

	//console.log(dirent.name + " : " + dirent.name.includes("true"));
	//addImage(classifier, mobilenet, fs.readFileSync(directory + dirent.name), dirent.name.includes("true"));
	*/

}

learn();

async function addImage(image, label) {
	const img0 = tf.browser.fromPixels(jpeg.decode(image));
	const logits0 = mobilenet.infer(img0, true);
	classifier.addExample(logits0, label);
}

async function testImage(image) {
	const x = tf.browser.fromPixels(jpeg.decode(image));
	const xlogits = mobilenet.infer(x, true);
	var result = await classifier.predictClass(xlogits)
	console.log(result);
}