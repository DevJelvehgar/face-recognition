#!/usr/bin/env python3
"""
Face Recognition Database Builder
This script builds a face database from images in the temp folder
"""

import os
import pickle
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from pathlib import Path

class FaceDatabaseBuilder:
    def __init__(self, temp_folder="temp", database_file="face_database.pkl"):
        self.temp_folder = temp_folder
        self.database_file = database_file
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device
        )
        
        # Initialize FaceNet model
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.face_database = {}
        
    def extract_face_embedding(self, image_path):
        """Extract face embedding from an image"""
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            
            # Detect face and extract embedding
            face_tensor = self.mtcnn(img)
            
            if face_tensor is not None:
                # Get face embedding
                with torch.no_grad():
                    face_embedding = self.facenet(face_tensor.unsqueeze(0).to(self.device))
                    return face_embedding.cpu().numpy().flatten()
            else:
                print(f"No face detected in {image_path}")
                return None
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def build_database(self):
        """Build face database from temp folder"""
        print("Building face database...")
        
        if not os.path.exists(self.temp_folder):
            print(f"Error: {self.temp_folder} folder not found!")
            return False
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for file_path in Path(self.temp_folder).rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                image_files.append(str(file_path))
        
        print(f"Found {len(image_files)} images")
        
        processed_count = 0
        successful_count = 0
        
        for image_path in image_files:
            print(f"Processing: {image_path}")
            
            # Extract face embedding
            embedding = self.extract_face_embedding(image_path)
            
            if embedding is not None:
                # Store in database
                filename = os.path.basename(image_path)
                self.face_database[filename] = {
                    'embedding': embedding,
                    'path': image_path
                }
                successful_count += 1
                print(f"✓ Successfully processed: {filename}")
            else:
                print(f"✗ Failed to process: {image_path}")
            
            processed_count += 1
        
        # Save database
        with open(self.database_file, 'wb') as f:
            pickle.dump(self.face_database, f)
        
        print(f"\nDatabase building completed!")
        print(f"Total images processed: {processed_count}")
        print(f"Successfully processed: {successful_count}")
        print(f"Database saved to: {self.database_file}")
        
        return True

def main():
    """Main function"""
    print("Face Recognition Database Builder")
    print("=" * 40)
    
    builder = FaceDatabaseBuilder()
    success = builder.build_database()
    
    if success:
        print("\n✓ Database built successfully!")
        print("You can now run the FastAPI server with: python main.py")
    else:
        print("\n✗ Database building failed!")

if __name__ == "__main__":
    main()
