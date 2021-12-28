const container = document.querySelector('#container');
const fileInput = document.querySelector('#file-input');

async function loadTranningData(){
	const labels = ['Đặng Quốc Trung', 'Đào Anh Thư', 'Diêm Công Hoàng', 'Đỗ Văn Huân', 'Nguyễn Thị Uyển', 'Phạm Đình Tân'];


	const labeledFaceDescriptors = [];
	for(const label of labels){
		const descriptions = [];
		for(let i = 1; i <= 4; i++){
			const image = await faceapi.fetchImage(`./data/${label}/${i}.jpg`);
			const detection = await faceapi.detectSingleFace(image).withFaceLandmarks().withFaceDescriptor();
			descriptions.push(detection.descriptor);
		}
		labeledFaceDescriptors.push(new faceapi.LabeledFaceDescriptors(label, descriptions));
		
		Toastify({text: `Tranning xong dữ liệu của ${label}`, duration: 1000}).showToast();
	}

	return labeledFaceDescriptors;
}

let faceMatcher 
async function init(){

	//Chờ dữ liệu được load xong
	await Promise.all([
		faceapi.nets.ssdMobilenetv1.loadFromUri('./models'),
		faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
		faceapi.nets.faceRecognitionNet.loadFromUri('./models')
	]);

	const tranningData = await loadTranningData();
	console.log(tranningData);
	//Độ lêch để nhận diện cho phép : 0.6 
	faceMatcher = new faceapi.FaceMatcher(tranningData, 0.6);
	//Hiển thị thông báo tải xong
	Toastify({text: 'Đã tải xong models nhận diện', duration: 1000}).showToast();
	document.querySelector("#loading").remove();
}

init();

fileInput.addEventListener('change', async (e) => {
	const file = fileInput.files[0];

	const image = await faceapi.bufferToImage(file);
	const canvas = faceapi.createCanvasFromMedia(image);


	container.innerHTML = '';
	container.append(image);
	container.append(canvas);

	const size = { width: image.width, height: image.height };
	//matchDemensions là hàm để điều chỉnh kích thước của canvas với kích thước của ảnh
	faceapi.matchDimensions(canvas, size);



	//FaceLandmarks68 là lưu các điểm trên khuôn mặt
	//FaceDescriptor là lưu các điểm trên khuôn mặt và lưu các điểm trên khuôn mặt
	const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors();
	const resizeDetections = faceapi.resizeResults(detections, size);

	for(const detection of resizeDetections){
		//Tạo 1 box với kích thước của khuôn mặt
		const box = detection.detection.box;
		//DrawBox là hàm để vẽ hình chữ nhật và hiển thị đó là khuôn mặt
		const drawBox = new faceapi.draw.DrawBox(box, { 
			label: faceMatcher.findBestMatch(detection.descriptor).toString(),
		});
		drawBox.draw(canvas);
	}
})