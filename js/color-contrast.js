(function () {
  var colorPicker = document.querySelector('#color-picker');
  var colorCode = document.querySelector('#color-code');
  var colorCodeHex = document.querySelector('#color-display > span');
  var body = document.querySelector('body');

  var textColors = ['black', 'white'];
  var backgrounds = [];
  var colors = [];

  var data = [
    { "input": { "r": "0.00", "g": "0.00", "b": "0.00" }, "output": { "white": 1 } },
    { "input": { "r": "0.85", "g": "0.88", "b": "0.56" }, "output": { "black": 1 } },
    { "input": { "r": "0.33", "g": "0.35", "b": "0.84" }, "output": { "white": 1 } },
    { "input": { "r": "0.00", "g": "0.99", "b": "1.00" }, "output": { "black": 1 } }
  ]

  for (var i = 0; i < data.length; i++) {
    backgrounds.push([data[i].input.r, data[i].input.g, data[i].input.b]);
    colors.push(textColors.indexOf(Object.keys(data[i].output)[0]))
  }

  var colorsTensor = tf.tensor1d(colors, 'int32')

  var xs = tf.tensor2d(backgrounds);
  var ys = tf.oneHot(colorsTensor, 2)

  var model = tf.sequential();

  var hidden = tf.layers.dense({
    units: 16,
    activation: 'sigmoid',
    inputShape: [3]
  });

  var output = tf.layers.dense({
    units: 2,
    activation: 'softmax'
  });

  model.add(hidden);
  model.add(output);

  var lr = 0.2;
  var optimizer = tf.train.sgd(lr);

  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy'
  });

  var options = {
    epochs: 50,
    shuffle: true,
    callbacks: {
      onTrainBegin() {},
      onTrainEnd() {},
      onEpochBegin() {},
      onEpochEnd() {},
      onBatchBegin() {},
      onBatchEnd() {}
    }
  }

  model.fit(xs, ys, options).then(function() {
    updateScreen(colorPicker.value);
  })

  var hex2rgb = function hex2rgb (hex) {
    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: Math.round(parseInt(result[1], 16) / 2.55) / 100,
      g: Math.round(parseInt(result[2], 16) / 2.55) / 100,
      b: Math.round(parseInt(result[3], 16) / 2.55) / 100
    } : null;
  }

  var updateScreen = function updateScreen(hex) {
    var rgb = hex2rgb(hex);
    var x = tf.tensor2d([
      [rgb.r, rgb.g, rgb.b]
    ])
  
    var result = model.predict(x); 
  
    result.argMax(1).data().then((index) => {
      var contrastColor = textColors[index[0]];
      body.style.backgroundColor = hex;
      colorCode.style.color = contrastColor;
      colorCode.style.borderColor = contrastColor;
      colorCode.value = hex.substr(1);
      colorCodeHex.style.color = hex;
      colorCodeHex.style.backgroundColor = contrastColor;
      colorPicker.value = hex;
    });
  }

  colorCode.value = colorPicker.value.substr(1);

  colorPicker.addEventListener('change', function (event) {
    var hex = event.target.value;
    updateScreen(hex);
  });

  colorCode.addEventListener('keyup', function (event) {
    var regex = /[0-9a-f]{6}|#[0-9a-f]{3}/gi;
    if (event.keyCode === 13 && 
        event.target.value.match(regex)) {
      updateScreen('#' + event.target.value);
    }
  });
})()