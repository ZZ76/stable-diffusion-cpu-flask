$(document).ready(function(){

	$("#width-range").on("input", function(){
		$("#width-display").text($("#width-range").val());
	});

	$("#height-range").on("input", function(){
		$("#height-display").text($("#height-range").val());
	});

	$.fn.loading_start = function(){
		$("#generate-button").prop("disabled", true);
		//$("#generated-image").hide();
		$("#spinner").show();
	}

	$.fn.loading_finished = function(){
		$("#generate-button").prop("disabled", false);
		//$("#generated-image").show();
		$("#spinner").hide();
	}

	$.fn.load_image = function(json){
		let string = json["image"];
		string = "data:image/png;base64, " + string;
		$("#generated-image").attr("src", string);
	}
	
	$("#generate-button").click(function(){
		let data = {};
		data["width"] = $("#width-range").val();
		data["height"] = $("#height-range").val();
		data["text"] =  $("#input-text").val() || "2000 years later";
		console.log(data);
		$.fn.loading_start();
		$.ajax({
			url: "/api/stable_diffusion_generate",
			data: data,
			type: "get",
			dataType: "json"
		})
		//.done(function(r){console.log("received:", r)})
		.done($.fn.loading_finished)
		.done(function(r){$.fn.load_image(r)})
		.fail($.fn.loading_finished)
	});

	$("#width-display").text($("#width-range").val());
	$("#height-display").text($("#height-range").val());

})
