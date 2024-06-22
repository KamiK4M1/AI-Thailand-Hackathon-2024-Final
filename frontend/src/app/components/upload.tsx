"use client"
import React, { useState } from 'react';
import axios from 'axios';
import styles from '../components/css/ArrowIndicator.module.css';
// import Navbar from '../components/Navbar';


const Upload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<FileList | null>(null);
  const [predict, setPredict] = useState<number | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSelectedFile(event.target.files || null);
    console.log(event.target.files);
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
        console.log(formData)
        const response = await axios.post('http://127.0.0.1:8000/predict', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
  
        setPredict(response.data.predict);
  
        // Wait for a short period before sending the next file
        formData.delete('file');
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
    } catch (error) {
      console.error('Error uploading file:', error);
    }
  };

  return (
    <div>
      {/* <Navbar/> */}
    <div className='hero min-h-[90vh] bg-base-20'>
      <div className='hero-content text-center'>
        <div className="max-w-md">
          
          <div className={styles.container}>
            <div className={`${styles.arrow} ${predict == 120 ? styles.glow : ''}`}>←</div>
            <div className={`${styles.circle} ${predict == 150 ? styles.glow : ''}`}>○</div>
            <div className={`${styles.arrow} ${predict == 110 ? styles.glow : ''}`}>→</div>
          </div>
          <div className=' mt-10'>
            <div>
              <input type="file" onChange={handleFileChange} multiple className='file-input file-input-bordered file-input-info w-full max-w-xs' />
            </div>
            <button onClick={handleUpload} className='btn btn-outline btn-primary mt-10'>Upload</button>
            <div className="mockup-code mt-12 font-black p-5">
              <pre><code>{predict !== null ? predict : ''}</code></pre>
            </div> 
            <h1></h1>
          </div>
        </div>
      </div>
    </div>
    </div>
  );
};

export default Upload;