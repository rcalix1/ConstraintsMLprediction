
async function runExample1() {
    
  const x = new Float32Array(4);
  x[0] = parseFloat(document.getElementById('box0c1').value) || 0;
  x[1] = parseFloat(document.getElementById('box1c1').value) || 0;
  x[2] = parseFloat(document.getElementById('box2c1').value) || 0;
  x[3] = parseFloat(document.getElementById('box3c1').value) || 0;

  const tensorX = new ort.Tensor('float32', x, [1, 4]);

  try {
    const session = await ort.InferenceSession.create("./resNet_Inverse_realData_new.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data; // Float32Array length 7

    // render here (output is in scope)
    const predictions = document.getElementById('predictions1');
    predictions.innerHTML = `
      <hr>Got an output Tensor:<br/>
      <table>
        <tr><td>i_h2i_rate</td>          <td id="c1td0">${output[0].toFixed(2)}</td></tr>
        <tr><td>i_h2_temp</td>           <td id="c1td1">${output[1].toFixed(2)}</td></tr>
        <tr><td>i_ngi_rate</td>          <td id="c1td2">${output[2].toFixed(2)}</td></tr>
        <tr><td>i_pci_rate</td>          <td id="c1td3">${output[3].toFixed(2)}</td></tr>
        <tr><td>i_o2_volfract</td>       <td id="c1td4">${output[4].toFixed(2)}</td></tr>
        <tr><td>i_hbtemp</td>            <td id="c1td5">${output[5].toFixed(2)}</td></tr>
        <tr><td>i_wind_rt</td>           <td id="c1td6">${output[6].toFixed(2)}</td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }

    runAvg();
}



async function runExampleLEforward() {
    
  const x = new Float32Array(7);
  x[0] = parseFloat(document.getElementById('LEbox0c1').value) || 0;
  x[1] = parseFloat(document.getElementById('LEbox1c1').value) || 0;
  x[2] = parseFloat(document.getElementById('LEbox2c1').value) || 0;
  x[3] = parseFloat(document.getElementById('LEbox3c1').value) || 0;
  x[4] = parseFloat(document.getElementById('LEbox4c1').value) || 0;
  x[5] = parseFloat(document.getElementById('LEbox5c1').value) || 0;
  x[6] = parseFloat(document.getElementById('LEbox6c1').value) || 0;  

  const tensorX = new ort.Tensor('float32', x, [1, 7]);

  try {
    const session = await ort.InferenceSession.create("./LEPINE_model_Forward.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data; // Float32Array length 4

    // render here (output is in scope)
    const predictions = document.getElementById('predictionsLEforward');
    predictions.innerHTML = `
      <hr>Got an output Tensor:<br/>
      <table>
        <tr><td>tgt</td>          <td id="c1td0">${output[0].toFixed(2)}</td></tr>
        <tr><td>hmt</td>           <td id="c1td1">${output[1].toFixed(2)}</td></tr>
        <tr><td>prod rate</td>          <td id="c1td2">${output[2].toFixed(2)}</td></tr>
        <tr><td>fta</td>          <td id="c1td3">${output[3].toFixed(2)}</td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }

    
}



async function runExampleLEinverse() {
    
  const x = new Float32Array(4);
  x[0] = parseFloat(document.getElementById('LEbox0c2').value) || 0;
  x[1] = parseFloat(document.getElementById('LEbox1c2').value) || 0;
  x[2] = parseFloat(document.getElementById('LEbox2c2').value) || 0;
  x[3] = parseFloat(document.getElementById('LEbox3c2').value) || 0;
   
  const tensorX = new ort.Tensor('float32', x, [1, 4]);

  try {
    const session = await ort.InferenceSession.create("./LEPINE_model_Inverse_2.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data; // Float32Array length 7

    // render here (output is in scope)
    const predictions = document.getElementById('predictionsLEinverse');
    predictions.innerHTML = `
      <hr>Got an output Tensor:<br/>
      <table>
        <tr><td>i_h2i_rate</td>          <td id="c1td0">${output[0].toFixed(2)}</td></tr>
        <tr><td>pci rate</td>           <td id="c1td1">${output[1].toFixed(2)}</td></tr>
        <tr><td>i_ngi_rate</td>          <td id="c1td2">${output[2].toFixed(2)}</td></tr>
        <tr><td>o2 vol fract</td>          <td id="c1td3">${output[3].toFixed(2)}</td></tr>
        <tr><td>h2 temp</td>           <td id="c1td1">${output[4].toFixed(2)}</td></tr>
        <tr><td> hb temp </td>          <td id="c1td2">${output[5].toFixed(2)}</td></tr>
        <tr><td> wind rate</td>          <td id="c1td3">${output[6].toFixed(2)}</td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }

    
}



async function runExample2() {
    
  const x = new Float32Array(4);
  x[0] = parseFloat(document.getElementById('box0c2').value) || 0;
  x[1] = parseFloat(document.getElementById('box1c2').value) || 0;
  x[2] = parseFloat(document.getElementById('box2c2').value) || 0;
  x[3] = parseFloat(document.getElementById('box3c2').value) || 0;

  const tensorX = new ort.Tensor('float32', x, [1, 4]);

  try {
    const session = await ort.InferenceSession.create("./resNet_Inverse_realData_new.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data; // Float32Array length 7

    // render here (output is in scope)
    const predictions = document.getElementById('predictions2');
    predictions.innerHTML = `
      <hr>Got an output Tensor:<br/>
      <table>
        <tr><td>i_h2i_rate</td>          <td id="c2td0">${output[0].toFixed(2)}</td></tr>
        <tr><td>i_h2_temp</td>           <td id="c2td1">${output[1].toFixed(2)}</td></tr>
        <tr><td>i_ngi_rate</td>          <td id="c2td2">${output[2].toFixed(2)}</td></tr>
        <tr><td>i_pci_rate</td>          <td id="c2td3">${output[3].toFixed(2)}</td></tr>
        <tr><td>i_o2_volfract</td>       <td id="c2td4">${output[4].toFixed(2)}</td></tr>
        <tr><td>i_hbtemp</td>            <td id="c2td5">${output[5].toFixed(2)}</td></tr>
        <tr><td>i_wind_rt</td>           <td id="c2td6">${output[6].toFixed(2)}</td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }

runAvg();
    
}






async function runExampleSynthetic() {
    
  const x = new Float32Array(8);
  x[0] = parseFloat(document.getElementById('box0c1').value) || 0;
  x[1] = parseFloat(document.getElementById('box1c1').value) || 0;
  x[2] = parseFloat(document.getElementById('box2c1').value) || 0;
  x[3] = parseFloat(document.getElementById('box3c1').value) || 0;
    
  x[4] = parseFloat(document.getElementById('box0c2').value) || 0;
  x[5] = parseFloat(document.getElementById('box1c2').value) || 0;
  x[6] = parseFloat(document.getElementById('box2c2').value) || 0;
  x[7] = parseFloat(document.getElementById('box3c2').value) || 0;

  const tensorX = new ort.Tensor('float32', x, [1, 8]);

  try {
    const session = await ort.InferenceSession.create("./resNet_Inverse_syntheticData.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data; // Float32Array length 8

    // render here (output is in scope)
    const predictions = document.getElementById('predSynthetic');
    predictions.innerHTML = `
      <hr>Got an output Tensor:<br/>
      <table>
        <tr><td>i_h2i_rate</td>          <td id="syntd0">${output[0].toFixed(2)}</td></tr>
        <tr><td>i_h2_temp</td>           <td id="syntd1">${output[1].toFixed(2)}</td></tr>
        <tr><td>i_ngi_rate</td>          <td id="syntd2">${output[2].toFixed(2)}</td></tr>
        <tr><td>i_pci_rate</td>          <td id="syntd3">${output[3].toFixed(2)}</td></tr>
        <tr><td>i_o2_volfract</td>       <td id="syntd4">${output[4].toFixed(2)}</td></tr>
        <tr><td>i_hbtemp</td>            <td id="syntd5">${output[5].toFixed(2)}</td></tr>
        <tr><td>i_wind_rt</td>           <td id="syntd6">${output[6].toFixed(2)}</td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }

runAvg();
    
}





async function runExampleRealGenHong() {
    await runExampleHW1(); 
    await runExampleHW2();
    runAvgHW();  
}




async function runAvg() {
    
    var c1td0 = parseFloat( document.getElementById('c1td0').innerHTML );
    var c1td1 = parseFloat( document.getElementById('c1td1').innerHTML );
    var c1td2 = parseFloat( document.getElementById('c1td2').innerHTML );
    var c1td3 = parseFloat( document.getElementById('c1td3').innerHTML );
    var c1td4 = parseFloat( document.getElementById('c1td4').innerHTML );
    var c1td5 = parseFloat( document.getElementById('c1td5').innerHTML );
    var c1td6 = parseFloat( document.getElementById('c1td6').innerHTML );
  
    
    var c2td0 = parseFloat( document.getElementById('c2td0').innerHTML );
    var c2td1 = parseFloat( document.getElementById('c2td1').innerHTML );
    var c2td2 = parseFloat( document.getElementById('c2td2').innerHTML );
    var c2td3 = parseFloat( document.getElementById('c2td3').innerHTML );
    var c2td4 = parseFloat( document.getElementById('c2td4').innerHTML );
    var c2td5 = parseFloat( document.getElementById('c2td5').innerHTML );
    var c2td6 = parseFloat( document.getElementById('c2td6').innerHTML );

    
    td0 = (c1td0 + c2td0)/2;
    td1 = (c1td1 + c2td1)/2;
    td2 = (c1td2 + c2td2)/2;
    td3 = (c1td3 + c2td3)/2;
    td4 = (c1td4 + c2td4)/2;
    td5 = (c1td5 + c2td5)/2;
    td6 = (c1td6 + c2td6)/2;

 
     difference.innerHTML = `<hr> Average is: <br/> 
 <table>
 
  <tr>
  <td> i_h2i_rate </td>
  <td> ${td0.toFixed(2)} </td>
  </tr>
  
  <tr>
  <td>  i_h2_temp </td>
  <td> ${td1.toFixed(2)} </td>
  </tr> 
  
  <tr>
  <td> i_ngi_rate  </td>
  <td> ${td2.toFixed(2)} </td>
  </tr> 

   <tr>
  <td> i_pci_rate</td>
  <td> ${td3.toFixed(2)} </td>
  </tr>
  
  <tr>
  <td>  i_o2_volfract </td>
  <td> ${td4.toFixed(2)} </td>
  </tr> 
  
  <tr>
  <td> i_hbtemp  </td>
  <td> ${td5.toFixed(2)} </td>
  </tr> 

   <tr>
  <td> i_wind_rt </td>
  <td> ${td6.toFixed(2)} </td>
  </tr> 
 
 </table>   `;
    
}


async function runExampleHW1() {
    
  const x = new Float32Array(4);
  x[0] = parseFloat(document.getElementById('box0c1').value) || 0;
  x[1] = parseFloat(document.getElementById('box1c1').value) || 0;
  x[2] = parseFloat(document.getElementById('box2c1').value) || 0;
  x[3] = parseFloat(document.getElementById('box3c1').value) || 0;

  const tensorX = new ort.Tensor('float32', x, [1, 4]);

  try {
    const session = await ort.InferenceSession.create("./F1F2_Inverse_HongRealGen.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data;    // Float32Array length 7

    // render here (output is in scope)
    const predictions = document.getElementById('predictionsHW1');
    predictions.innerHTML = `
      <hr>Got an output Tensor:<br/>
      <table>
        <tr><td>i_h2i_rate</td>          <td id="c1td0HW">${output[0].toFixed(2)}</td></tr>
        <tr><td>i_h2_temp</td>           <td id="c1td1HW">${output[1].toFixed(2)}</td></tr>
        <tr><td>i_ngi_rate</td>          <td id="c1td2HW">${output[2].toFixed(2)}</td></tr>
        <tr><td>i_pci_rate</td>          <td id="c1td3HW">${output[3].toFixed(2)}</td></tr>
        <tr><td>i_o2_volfract</td>       <td id="c1td4HW">${output[4].toFixed(2)}</td></tr>
        <tr><td>i_hbtemp</td>            <td id="c1td5HW">${output[5].toFixed(2)}</td></tr>
        <tr><td>i_wind_rt</td>           <td id="c1td6HW">${output[6].toFixed(2)}</td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }

    runAvgHW();
}



async function runExampleHW2() {
    
  const x = new Float32Array(4);
  x[0] = parseFloat(document.getElementById('box0c2').value) || 0;
  x[1] = parseFloat(document.getElementById('box1c2').value) || 0;
  x[2] = parseFloat(document.getElementById('box2c2').value) || 0;
  x[3] = parseFloat(document.getElementById('box3c2').value) || 0;

  const tensorX = new ort.Tensor('float32', x, [1, 4]);

  try {
    const session = await ort.InferenceSession.create("./F1F2_Inverse_HongRealGen.onnx?v=" + Date.now());
    const results = await session.run({ input1: tensorX });
    const output = results.output1.data; // Float32Array length 7

    // render here (output is in scope)
    const predictions = document.getElementById('predictionsHW2');
    predictions.innerHTML = `
      <hr>Got an output Tensor:<br/>
      <table>
        <tr><td>i_h2i_rate</td>          <td id="c2td0HW">${output[0].toFixed(2)}</td></tr>
        <tr><td>i_h2_temp</td>           <td id="c2td1HW">${output[1].toFixed(2)}</td></tr>
        <tr><td>i_ngi_rate</td>          <td id="c2td2HW">${output[2].toFixed(2)}</td></tr>
        <tr><td>i_pci_rate</td>          <td id="c2td3HW">${output[3].toFixed(2)}</td></tr>
        <tr><td>i_o2_volfract</td>       <td id="c2td4HW">${output[4].toFixed(2)}</td></tr>
        <tr><td>i_hbtemp</td>            <td id="c2td5HW">${output[5].toFixed(2)}</td></tr>
        <tr><td>i_wind_rt</td>           <td id="c2td6HW">${output[6].toFixed(2)}</td></tr>
      </table>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }

runAvgHW();
    
}



async function runAvgHW() {
    
    var c1td0 = parseFloat( document.getElementById('c1td0HW').innerHTML );
    var c1td1 = parseFloat( document.getElementById('c1td1HW').innerHTML );
    var c1td2 = parseFloat( document.getElementById('c1td2HW').innerHTML );
    var c1td3 = parseFloat( document.getElementById('c1td3HW').innerHTML );
    var c1td4 = parseFloat( document.getElementById('c1td4HW').innerHTML );
    var c1td5 = parseFloat( document.getElementById('c1td5HW').innerHTML );
    var c1td6 = parseFloat( document.getElementById('c1td6HW').innerHTML );
  
    
    var c2td0 = parseFloat( document.getElementById('c2td0HW').innerHTML );
    var c2td1 = parseFloat( document.getElementById('c2td1HW').innerHTML );
    var c2td2 = parseFloat( document.getElementById('c2td2HW').innerHTML );
    var c2td3 = parseFloat( document.getElementById('c2td3HW').innerHTML );
    var c2td4 = parseFloat( document.getElementById('c2td4HW').innerHTML );
    var c2td5 = parseFloat( document.getElementById('c2td5HW').innerHTML );
    var c2td6 = parseFloat( document.getElementById('c2td6HW').innerHTML );

    
    td0 = (c1td0 + c2td0)/2;
    td1 = (c1td1 + c2td1)/2;
    td2 = (c1td2 + c2td2)/2;
    td3 = (c1td3 + c2td3)/2;
    td4 = (c1td4 + c2td4)/2;
    td5 = (c1td5 + c2td5)/2;
    td6 = (c1td6 + c2td6)/2;

 
     differenceHW.innerHTML = `<hr> Average is: <br/> 
 <table>
 
  <tr>
  <td> i_h2i_rate </td>
  <td> ${td0.toFixed(2)} </td>
  </tr>
  
  <tr>
  <td>  i_h2_temp </td>
  <td> ${td1.toFixed(2)} </td>
  </tr> 
  
  <tr>
  <td> i_ngi_rate  </td>
  <td> ${td2.toFixed(2)} </td>
  </tr> 

   <tr>
  <td> i_pci_rate</td>
  <td> ${td3.toFixed(2)} </td>
  </tr>
  
  <tr>
  <td>  i_o2_volfract </td>
  <td> ${td4.toFixed(2)} </td>
  </tr> 
  
  <tr>
  <td> i_hbtemp  </td>
  <td> ${td5.toFixed(2)} </td>
  </tr> 

   <tr>
  <td> i_wind_rt </td>
  <td> ${td6.toFixed(2)} </td>
  </tr> 
 
 </table>   `;
    
}

