"use client"
import React, { useState } from 'react';
import axios from 'axios';
import styles from '../components/css/ArrowIndicator.module.css';
// import Navbar from '../components/Navbar';

const Upload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<FileList | null>(null);
  const [predict, setPredict] = useState<string | null>(null);
  const [imageUrls, setImageUrls] = useState<string[]>([]);
  const [userInput, setUserInput] = useState<string>('');

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      setSelectedFile(files);
      const urls = Array.from(files).map(file => URL.createObjectURL(file));
      setImageUrls(urls);
    }
  };

  const handleTextInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setUserInput(event.target.value);
  };

  const handleUpload = async () => {
    try {
      if (!selectedFile || selectedFile.length === 0) {
        console.error('No files selected');
        return;
      }

      for (let i = 0; i < selectedFile.length; i++) {
        const formData = new FormData();
        formData.append('file', selectedFile[i]);
        console.log(formData);
        const response = await axios.post('https://api-obon.conf.in.th/team13/image-to-text', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });

        setPredict(response.data.description);

        // Wait for a short period before sending the next file
        formData.delete('file');
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  const handleReupload = async () => {
    try {
      if (!selectedFile || selectedFile.length === 0) {
        console.error('No files selected');
        return;
      }

      for (let i = 0; i < selectedFile.length; i++) {
        const formData = new FormData();
        formData.append('file', selectedFile[i]);
        formData.append('text', userInput);
        console.log(formData);
        await axios.post('https://your-second-upload-endpoint', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });

        // Wait for a short period before sending the next file
        formData.delete('file');
        formData.delete('text');
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
    } catch (error) {
      console.error('Error re-uploading file:', error);
    }
  };

  return (
    <div>
      {/* <Navbar/> */}
      <div className='hero min-h-[100vh] bg-base-20'>
        <div className='hero-content text-center'>
          <div className="max-w-md">
            <div className='mt-10'>
              <div>
                <input 
                  type="file" 
                  onChange={handleFileChange} 
                  multiple 
                  className='file-input file-input-bordered file-input-info w-full max-w-xs' 
                />
              </div>
              <button 
                onClick={handleUpload} 
                className='btn btn-outline btn-primary mt-10'
              >
                Upload
              </button>
              <div className="mt-10">
                {imageUrls.map((url, index) => (
                  <div key={index} className="mb-4 text-center">
                    <img src={url} alt={`Preview ${index + 1}`} className="max-w-xs mx-auto" />
                  </div>
                ))}
              </div>
              <div className="mockup-code mt-12 font-black p-5">
                <pre><code>{predict !== null ? predict : ''}</code></pre>
              </div>
              {predict && (
                <div className='mt-10'>
                  <input 
                    type="text" 
                    value={userInput} 
                    onChange={handleTextInputChange} 
                    className='input input-bordered input-info w-full max-w-xs' 
                    placeholder="Enter additional text" 
                  />
                  <button 
                    onClick={handleReupload} 
                    className='btn btn-outline btn-primary mt-10'
                  >
                    Re-upload with Text
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Upload;
