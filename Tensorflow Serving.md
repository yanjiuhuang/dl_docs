# Tensorflow Serving

## Basics
我们主要使用Tensorflow Serving来导出模型，并使用`tensorflow_model_server`来提供serving的能力。

### 训练和导出Tensorflow模型
对于训练好的模型，我们可以使用Tensorflow Serving `Exporter`模块来导出。`Exporter`保存了一个训练完的模型的快照(snapshot)。这样就能通过保存到一个可靠存储(例如disk)中用于后续的推断。

	from tensorflow.contrib.session_bundle import exporter
	...
	export_path = sys.argv[-1]
	print 'Exporting trained model to', export_path
	saver = tf.train.Saver(sharded=True)
	model_exporter = exporter.Exporter(saver)
	model_exporter.init(
	    sess.graph.as_graph_def(),
	    named_graph_signatures={
	        'inputs': exporter.generic_signature({'images': x}),
	        'outputs': exporter.generic_signature({'scores': y})})
	model_exporter.export(export_path, tf.constant(FLAGS.export_version), sess)  
	
首先`saver`会将graph中的变量序列化到模型导出中，这样后续才能正确地恢复。这里由于没有指定`variable_list`，所以默认就会导出图中的所有变量。对于更复杂的图，我们可以选择导出只是用于推断的变量即可。  

在`Exporter.init()`时，使用了如下的参数:

* `sess.graph.as_graph_def()`是图的protobuf形式。`export`将对这个protobuf导出。
* `named_graph_signatures=...`指定了一个导出模型的签名。这个签名中指定了导出模型的类型，以及绑定到模型用于进行推断的input/output tensors。在上述例子中，使用了标准`tensorflow_model_server`支持的`exporter.generic_signature`，其中`outputs`和`inputs`作为了它的key。  

	* 上述签名中使用的`images`和`scores`都是tensor的别名。我们只要保证其唯一性，那么他们就会成为`x`，`y`的逻辑名(logical names)。如果logical names和real names不一致，那么就会建议一个logical names到real names的映射。  

在完成了模型导出之后，我们就可以使用Standard Tensorflow Model Server来加载模型，提供serving了：  

	$>bazel build //tensorflow_serving/model_servers:tensorflow_model_server
	$>bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/  
	
Model Server启动之后，我们就可以通过client来进行推断了。client端与serving之间是基于gRPC。简单的示例代码如下所示：

	from grpc.beta import implementations
	import tensorflow as tf

	from tensorflow_serving.apis import predict_pb2
	from tensorflow_serving.apis import prediction_service_pb2
	
	host, port = FLAGS.server.split(':')
	channel = implementations.insecure_channel(host, int(port))
	stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
	  # Send request
	  with open(FLAGS.image, 'rb') as f:
	    # See prediction_service.proto for gRPC request/response details.
	    data = f.read()
	    request = predict_pb2.PredictRequest()
	    request.model_spec.name = 'inception'
	    request.inputs['images'].CopyFrom(
	        tf.contrib.util.make_tensor_proto(data, shape=[1]))
	    result = stub.Predict(request, 10.0)  # 10 secs timeout
	    print(result)
	    
	    
## 构建标准TensorFlow Model Server
