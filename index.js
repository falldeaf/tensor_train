const Canvas = require('canvas');
const fs = require('fs');
const jpeg = require('jpeg-js');
const tf = require('@tensorflow/tfjs');
const mobilenetModule = require('@tensorflow-models/mobilenet');
const knnClassifier = require('@tensorflow-models/knn-classifier');

let Client = require('ssh2-sftp-client');
let sftp = new Client();

const directory = 'images/plant/';

const init = async function() {

	console.log(process.env.HOST);

	/*
	sftp.connect({
		host: process.env.HOST,
		port: '8080',
		username: 'username',
		password: '******'
	}).then(() => {
		return sftp.list('/pathname');
	}).then(data => {
		console.log(data, 'the data info');
	}).catch(err => {
		console.log(err, 'catch error');
	});

	
	const classifier = knnClassifier.create();
	const mobilenet = await mobilenetModule.load();

	await addImage(classifier, mobilenet, fs.readFileSync('images/plant/plant_false (2).jpg'), "fine");
	await addImage(classifier, mobilenet, fs.readFileSync('images/plant/plant_false (3).jpg'), "fine");
	await addImage(classifier, mobilenet, fs.readFileSync('images/plant/plant_false (4).jpg'), "fine");

	await addImage(classifier, mobilenet, fs.readFileSync('images/plant/plant_true (3).jpg'), "dry");
	await addImage(classifier, mobilenet, fs.readFileSync('images/plant/plant_true (4).jpg'), "dry");
	await addImage(classifier, mobilenet, fs.readFileSync('images/plant/plant_true.jpg'), "dry");

	//await testImage(classifier, mobilenet, fs.readFileSync('images/plant/test_false.jpg'));

	console.log(await classifier.getClassifierDataset());

	//console.log(dirent.name + " : " + dirent.name.includes("true"));
	//addImage(classifier, mobilenet, fs.readFileSync(directory + dirent.name), dirent.name.includes("true"));
	*/

}

init();

async function addImage(classifier, mobilenet, image, label) {
	const img0 = tf.browser.fromPixels(jpeg.decode(image));
	const logits0 = mobilenet.infer(img0, true);
	classifier.addExample(logits0, label);
}

async function testImage(classifier, mobilenet, image) {
	const x = tf.browser.fromPixels(jpeg.decode(image));
	const xlogits = mobilenet.infer(x, true);
	var result = await classifier.predictClass(xlogits)
	console.log(result);
}