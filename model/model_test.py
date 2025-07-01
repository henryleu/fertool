#!/usr/bin/env python3
"""
ONNX Model Shape Testing Script
Tests the input and output shapes of the yolo.onnx model file
"""

import onnx
import numpy as np
import os
import sys

def test_model_shape(model_path):
    """Test and display the input/output shapes of an ONNX model"""
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    try:
        # Load the ONNX model
        model = onnx.load(model_path)
        
        print(f"Testing model: {model_path}")
        print("=" * 50)
        
        # Check if model is valid
        onnx.checker.check_model(model)
        print("✓ Model validation passed")
        
        # Get model graph
        graph = model.graph
        
        # Display input information
        print("\n📥 INPUT INFORMATION:")
        print("-" * 30)
        for i, input_tensor in enumerate(graph.input):
            print(f"Input {i+1}: {input_tensor.name}")
            
            # Get shape information
            shape = []
            for dim in input_tensor.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append("?")
            
            print(f"  Shape: {shape}")
            print(f"  Data type: {onnx.TensorProto.DataType.Name(input_tensor.type.tensor_type.elem_type)}")
        
        # Display output information
        print("\n📤 OUTPUT INFORMATION:")
        print("-" * 30)
        for i, output_tensor in enumerate(graph.output):
            print(f"Output {i+1}: {output_tensor.name}")
            
            # Get shape information
            shape = []
            for dim in output_tensor.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append("?")
            
            print(f"  Shape: {shape}")
            print(f"  Data type: {onnx.TensorProto.DataType.Name(output_tensor.type.tensor_type.elem_type)}")
        
        # Display general model information
        print("\n📋 MODEL INFORMATION:")
        print("-" * 30)
        print(f"Model version: {model.model_version}")
        print(f"IR version: {model.ir_version}")
        print(f"Producer name: {model.producer_name}")
        print(f"Producer version: {model.producer_version}")
        print(f"Number of nodes: {len(graph.node)}")
        
        return True
        
    except Exception as e:
        print(f"Error loading or processing model: {str(e)}")
        return False

def main():
    """Main function to run the model shape test"""
    
    # Default model path
    model_path = "yolo.onnx"
    
    # Check if model path is provided as command line argument
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    # Make sure we're looking in the current directory (.model)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_model_path = os.path.join(script_dir, model_path)
    
    print("🔍 ONNX Model Shape Tester")
    print("=" * 50)
    
    success = test_model_shape(full_model_path)
    
    if success:
        print("\n✅ Model shape testing completed successfully!")
    else:
        print("\n❌ Model shape testing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()