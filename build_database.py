#!/usr/bin/env python3
"""
Face Recognition Database Builder
This script builds a face database from images in the E:\\images folder and its subdirectories
"""

import os
import pickle
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from pathlib import Path
import re
import glob

class FaceDatabaseBuilder:
    def __init__(self, images_folder=r"E:\images", database_file="face_database.pkl"):
        self.images_folder = images_folder
        self.database_file = database_file
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(
            image_size=160, # FaceNet model input size
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], # [face detection, face refinement , facial landmark detection]
            factor=0.709,
            post_process=True,
            device=self.device
        )
        
        # Initialize FaceNet model
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.face_database = {}
        
    def extract_face_embedding(self, image_path):
        """Extract face embedding from an image"""
        """
        Image File → Load → Convert to RGB → Detect Face → Generate Embedding → Return Array
              ↓        ↓         ↓             ↓                     ↓              ↓
        "photo.jpg" → PIL → RGB Image →      MTCNN → Face Tensor → FaceNet → [512 numbers]
        """
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            
            # Detect face and extract embedding
            face_tensor = self.mtcnn(img)
            
            if face_tensor is not None:
                # Get face embedding - not need to compute gradients
                with torch.no_grad():
                    face_embedding = self.facenet(face_tensor.unsqueeze(0).to(self.device))
                    # return 512-dimensional face embedding and flatten it to 1D array
                    return face_embedding.cpu().numpy().flatten()
            else:
                print(f"No face detected in {image_path}")
                return None
                
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def build_database(self):
        """Build face database from images folder"""
        print("Building face database...")
        
        if not os.path.exists(self.images_folder):
            print(f"Error: {self.images_folder} folder not found!")
            return False
        
        # Regex pattern to find all image files recursively
        # This pattern matches common image extensions in any subdirectory
        image_pattern = re.compile(r'.*\.(jpg|jpeg|png|bmp|tiff|gif|webp)$', re.IGNORECASE)
        
        
        # Use glob to get all files recursively
        """
            Example of image files (E:\images\**\*.jpg):
            E:\images\photo1.jpg
            E:\images\subfolder1\photo2.png
            E:\images\subfolder1\subfolder2\photo3.jpeg
            E:\images\another_folder\image.PNG
        """
        all_files = glob.glob(os.path.join(self.images_folder, '**', '*'), recursive=True)
        
        # Filter using regex pattern

        image_files = []
        for file_path in all_files:
            if os.path.isfile(file_path) and image_pattern.match(file_path):
                image_files.append(file_path)

        
        print(f"Found {len(image_files)} images in {self.images_folder} and its subdirectories")
        
        processed_count = 0
        successful_count = 0
        
        for image_path in image_files:
            print(f"Processing: {image_path}")
            
            # Extract face embedding return 512-dimensional face embedding and flatten it to 1D array
            embedding = self.extract_face_embedding(image_path)
            
            if embedding is not None:
                # Store in database
                # Extract filename from image path E:\images\folder\photo.jpg → photo.jpg
                filename = os.path.basename(image_path)
                self.face_database[filename] = {
                    'embedding': embedding, # 512-dimensional face embedding
                    'path': image_path # E:\images\folder\photo.jpg
                }
                successful_count += 1
                print(f"✓ Successfully processed: {filename}")
            else:
                print(f"✗ Failed to process: {image_path}")
            
            processed_count += 1
        
        # Save database -> write binary to file
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
