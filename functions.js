var parameters = {
    weight_min: 0.0,
    weight_max: 0.0,
    height_min: 0.0,
    height_max: 0.0,
    chest_min: 0.0,
    chest_max: 0.0,
    hip_min: 0.0,
    hip_max: 0.0,
    waist_min: 0.0,
    waist_max: 0.0,
    gender: 0
}

var session_female, session_male;
    
function gaussianRandom(mean=0, stdev=1) {
          const u = 1 - Math.random(); // Converting [0,1) to (0,1]
          const v = Math.random();
          const z = Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
          // Transform to the desired mean and standard deviation:
          return z * stdev + mean;
      }

async function create_session() {        
        session_female = await ort.InferenceSession.create('./vae_female.onnx');
        session_male = await ort.InferenceSession.create('./vae_male.onnx');
        await change_gender(0);
      }

async function run_inference() {

        batch = 8
        tmp = []
        for(let i = 0; i<batch*32; i++) {tmp.push(gaussianRandom())}
        const z = Float32Array.from(tmp);
                
        var weight = document.getElementById('weight_label').innerHTML;
        var height = document.getElementById('height_label').innerHTML;
        var chest = document.getElementById('chest_label').innerHTML;
        var hip = document.getElementById('hip_label').innerHTML;
        var waist = document.getElementById('waist_label').innerHTML;

        tmp = []        
        for(let i = 0; i<batch; i++) {
            tmp.push((weight - parameters.weight_min) / (parameters.weight_max - parameters.weight_min));
            tmp.push((height - parameters.height_min) / (parameters.height_max - parameters.height_min));
            tmp.push((chest - parameters.chest_min) / (parameters.chest_max - parameters.chest_min));
            tmp.push((hip - parameters.hip_min) / (parameters.hip_max - parameters.hip_min));
            tmp.push((waist - parameters.waist_min) / (parameters.waist_max - parameters.waist_min));
        }
        const meas = Float32Array.from(tmp);        
        
        const tensor_z = new ort.Tensor('float32', z, [batch, 32])
        const tensor_meas = new ort.Tensor('float32', meas, [batch, 5])

        const feeds = { "z": tensor_z, "meas": tensor_meas };

        // Run model depending on the gender
        result = await (parameters.gender ? session_male.run(feeds) : session_female.run(feeds))                
        
        dataC = result['output'].data;

        dataC_grey = []

        for (let i = 0; i < dataC.length; i += 1) {          
          dataC_grey.push(dataC[i]); // red
          dataC_grey.push(dataC[i]); // green
          dataC_grey.push(dataC[i]); // blue
          dataC_grey.push(255);
        }

        var canvas = document.getElementById('tutorial');
        var ctx = canvas.getContext('2d');
        
        
        // create imageData object
        var idata = ctx.createImageData(512, 512);

        // set our buffer as source
        idata.data.set(new Uint8ClampedArray(dataC_grey));

        // update canvas with new data
        ctx.putImageData(idata, 0, 0);

      }      


     function change_weight(value) {
        document.getElementById('weight_label').innerHTML = value;        
        run_inference();
    }

    function change_height(value) {
        document.getElementById('height_label').innerHTML = value;        
        run_inference();
    }

    function change_chest(value) {
        document.getElementById('chest_label').innerHTML = value;        
        run_inference();
    }

    function change_hip(value) {
        document.getElementById('hip_label').innerHTML = value;        
        run_inference();
    }

    function change_waist(value) {
        document.getElementById('waist_label').innerHTML = value;        
        run_inference();
    }

    function change_gender(selectedObject) {        
        
        if (selectedObject == 0) {
            document.getElementById("weightstr").min = 40.0;
            document.getElementById("heightstr").min = 1.40;
            document.getElementById("cheststr").min = 0.70;
            document.getElementById("hipstr").min = 0.81;
            document.getElementById("waiststr").min = 0.60;

            document.getElementById("weightstr").max = 176.0;
            document.getElementById("heightstr").max = 1.90;
            document.getElementById("cheststr").max = 1.94;
            document.getElementById("hipstr").max = 1.75;
            document.getElementById("waiststr").max = 1.83;

            parameters.weight_min = 40.0;
            parameters.weight_max = 176.0;
            parameters.height_min = 1.40;
            parameters.height_max = 1.90;
            parameters.chest_min = 0.70;
            parameters.chest_max = 1.94;
            parameters.hip_min = 0.81;
            parameters.hip_max = 1.75;
            parameters.waist_min = 0.60;
            parameters.waist_max = 1.83;

            parameters.gender = 0;

            run_inference();
        }

        if (selectedObject == 1) {
            document.getElementById("weightstr").min = 45.0;
            document.getElementById("heightstr").min = 1.46;
            document.getElementById("cheststr").min = 0.68;
            document.getElementById("hipstr").min = 0.81;
            document.getElementById("waiststr").min = 0.65;

            document.getElementById("weightstr").max = 213.0;
            document.getElementById("heightstr").max = 2.00;
            document.getElementById("cheststr").max = 2.38;
            document.getElementById("hipstr").max = 1.82;
            document.getElementById("waiststr").max = 1.79;

            parameters.weight_min = 45.0;
            parameters.weight_max = 213.0;
            parameters.height_min = 1.46;
            parameters.height_max = 2.00;
            parameters.chest_min = 0.68;
            parameters.chest_max = 2.38;
            parameters.hip_min = 0.81;
            parameters.hip_max = 1.82;
            parameters.waist_min = 0.65;
            parameters.waist_max = 1.79;

            parameters.gender = 1;

            run_inference();
        }
    }