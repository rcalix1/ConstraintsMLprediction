
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
}




async function runDiff() {
    
    var c1td0 = parseFloat( document.getElementById('c1td0').innerHTML );
    var c1td1 = parseFloat( document.getElementById('c1td1').innerHTML );
    var c1td2 = parseFloat( document.getElementById('c1td2').innerHTML );
  
    
    var c2td0 = parseFloat( document.getElementById('c2td0').innerHTML );
    var c2td1 = parseFloat( document.getElementById('c2td1').innerHTML );
    var c2td2 = parseFloat( document.getElementById('c2td2').innerHTML );

    
    td0 = c1td0 - c2td0;
    td1 = c1td1 - c2td1;
    td2 = c1td2 - c2td2;
  
 
     difference.innerHTML = `<hr> Difference is: <br/> 
 <table>
 
  <tr>
  <td> o_raceway_coal_burn_perce </td>
  <td> ${td0.toFixed(2)} </td>
  </tr>
  
  <tr>
  <td>  o_raceway_flame_temp_k </td>
  <td> ${td1.toFixed(2)} </td>
  </tr> 
  
  <tr>
  <td> o_raceway_volume_m  </td>
  <td> ${td2.toFixed(2)} </td>
  </tr> 
 
 </table>   `;
    
}

