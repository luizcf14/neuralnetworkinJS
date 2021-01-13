/****
 * Code developed by Luiz Cortinhas 
 * 
 */
var MLP = function(weight, layer_size, activation_functions, loss_function, learning_rate){
	this.weight = weight;
	this.activation_functions = activation_functions;
	this.layer_size = layer_size;
	this.learning_rate = learning_rate;
	this.loss_function = loss_function;
};
MLP.prototype.train = function(x, y, epochs){
	// full training
	for(var epoch=0; epoch<epochs; epoch++){
		// train all sets
		for(var training_set=0; training_set<x.length; training_set++){
			// update states
			var state = [];
			state[0] = Object.assign([], x[training_set]); // input layer
			for(var i_layer=0; i_layer<this.layer_size.length-1; i_layer++){
				var new_state = feed_forward(state[i_layer], this.weight[i_layer], this.layer_size[i_layer+1], this.activation_functions[i_layer]);
				state[i_layer+1] = new_state.concat();
			}
			// back-propagation-----------------------------------------------
			this.learning_rate = lr_annealing(epoch, this.learning_rate);
			//first bp
			var BP1 = back_propagation(state[state.length-2], state[state.length-1], this.learning_rate, this.loss_function, y[training_set], false);
			var deltaW = []; deltaW[this.layer_size.length-2] = BP1[0].concat();//delta-Weight Table
			var S = BP1[1].concat();
			//rest of the bps
			for(var i_state=state.length-2; i_state>0; i_state--){
				//calculate sum(w*S)
				var next_S = [];
				for(var from=0; from<this.layer_size[i_state]; from++){
					var summ = 0.0;
					for(var to=0; to<this.layer_size[i_state+1]; to++){
						summ += S[to]*this.weight[i_state][from][to];
					}
					next_S.push(summ);
				}
				//start this bp
				var BPs = back_propagation(state[i_state-1], state[i_state], this.learning_rate, this.loss_function, false ,next_S);
				deltaW[i_state-1] = BPs[0].concat();
				S = BPs[1].concat();
			}
			// update weights
			for(var i_weight=0; i_weight<this.weight.length; i_weight++){
				for(var from=0; from<this.layer_size[i_weight]; from++){
					for(var to=0; to<this.layer_size[i_weight+1]; to++){
						this.weight[i_weight][from][to] += deltaW[i_weight][from][to];
					}
				}
			}
			// end back-propagation-----------------------------------------------
		}
	}
};
MLP.prototype.predict = function(x){
	// update states
	var output = x.concat(); // input layer
	for(var i_layer=0; i_layer<this.layer_size.length-1; i_layer++){
		output = feed_forward(output, this.weight[i_layer], this.layer_size[i_layer+1], this.activation_functions[i_layer]);
	}
	return output;
}


function initialize_weights(sizes, min_n, max_n){
	var W = [];
	console.log('initializing random weights');
	var depth = sizes.length;
	for(var l=1; l<depth; l++){
		var _W = [];
		for(var i=0; i<sizes[l-1]; i++){
			_W[i] = [];
			for(var j=0; j<sizes[l]; j++){
				_W[i][j] = Math.random() * (max_n - min_n) + min_n;
			}
		}
		W[l-1] = _W.concat();
	}
	
	return W.concat();
}


function feed_forward(x, w, output_size, activation_function){
	if(activation_function=='sigmoid'){
		var activations = [];
		for(var j=0; j<output_size; j++){
			var summ = 0.0;
			for(var i=0; i<x.length; i++){
				summ += x[i]*w[i][j];
			}
			//summ += -1.0; //bias
			console.log('feed, bias disabled');
			console.log('feed='+summ)
			console.log('sigmoid(feed)='+sigmoid(summ))
			activations.push(sigmoid(summ));
		}
		return activations;
	} else if(activation_function=='softmax'){ //todo
		var activations = [];
		for(var j=0; j<output_size; j++){
			var summ = 0.0;
			for(var i=0; i<x.length; i++){
				summ += x[i]*w[i][j];
			}
			activations.push(summ);
		}
		activations = softmax(activations);
		return activations.concat();
	}
}


function sigmoid(a){
	return 1/(1+Math.exp(-a));
}
function softmax(a){
	var exp_sum = 0.0;
	for(var i_a=0; i_a<a.length; i_a++){
		exp_sum += Math.exp(a[i_a]);
	}
	for(var i_a=0; i_a<a.length; i_a++){
		a[i_a] = Math.exp(a[i_a])/exp_sum;
	}
	return a.concat()
}


function back_propagation(i, o, alpha, loss, d, sw){
	var S_array = [];
	var deltaW = [];
	if(d){ // first-bp layer
		for(var m=0; m<d.length; m++){
			// calculate a(x)'
			var S = 0.0;
			if(loss=='mse'){
				S = o[m]*(1-o[m])*(d[m]-o[m]);
			} else if(loss=='cross_entropy') {
				S = o[m]*(1-o[m])*(d[m]-o[m]); //todo
			}
			
			for(var n=0; n<i.length; n++){
				if(m==0) deltaW[n] = [];
				
				var dw = alpha*S*i[n];
				deltaW[n][m] = dw;
			}
			S_array.push(S);
		}
	} else {
		for(var m=0; m<o.length; m++){
			var S = o[m]*(1-o[m])*sw[m];
			for(var n=0; n<i.length; n++){
				if(m==0) deltaW[n] = [];
				
				var dw = alpha*S*i[n];
				deltaW[n][m] = dw;
			}
			S_array.push(S);
		}
	}
	
	return [ deltaW, S_array ];
}


function lr_annealing(epoch, lr, decay=0.0){
	return lr/(1+decay*epoch);
}














