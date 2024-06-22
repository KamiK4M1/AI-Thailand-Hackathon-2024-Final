"use client";
import React, { useState } from "react";
import axios from "axios";

const Upload: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<FileList | null>(null);
  const [predict, setPredict] = useState<string | null>(null);
  const [imageUrls, setImageUrls] = useState<string[]>([]);
  const [secondApiOutput, setSecondApiOutput] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      setSelectedFile(files);
      const urls = Array.from(files).map((file) => URL.createObjectURL(file));
      setImageUrls(urls);
    }
  };

  const handleUpload = async () => {
    try {
      if (!selectedFile || selectedFile.length === 0) {
        console.error("No files selected");
        return;
      }

      for (let i = 0; i < selectedFile.length; i++) {
        const formData = new FormData();
        formData.append("file", selectedFile[i]);
        console.log(formData);
        const response = await axios.post(
          "https://api-obon.conf.in.th/team13/image-to-text",
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
            },
          }
        );

        setPredict(response.data.description);

        // Wait for a short period before sending the next file
        formData.delete("file");
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
    } catch (error) {
      console.error("Error uploading file:", error);
    }
  };

  const handleReupload = async () => {
    try {
      if (!predict) {
        console.error("No prediction available");
        return;
      }

      const response = await axios.post(
        "https://api-obon.conf.in.th/team13/llm",
        null,
        {
          params: {
            text: predict,
          },
          responseType: "stream",
        }
      );
      const reader = response.data.getReader();

      let streamingText = "";

      const decodeStream = async () => {
        const { value, done } = await reader.read();
        if (done) {
          console.log("Stream complete");
          return;
        }
        streamingText += new TextDecoder("utf-8").decode(value);
        console.log("Stream chunk received:", streamingText);
        // Update state or handle streamingText as needed
        setSecondApiOutput(streamingText);
        console.log(streamingText)

        // Continue reading next chunk
        await decodeStream();
      };

      await decodeStream();
    } catch (error) {
      console.error("Error re-uploading file:", error);
    }
  };

  return (
    <div>
      <div className="hero min-h-[100vh] bg-base-20">
        <div className="hero-content text-center">
          <div className="max-w-md">
            <div className="mt-10">
              <div>
                <input
                  type="file"
                  onChange={handleFileChange}
                  multiple
                  className="file-input file-input-bordered file-input-info w-full max-w-xs"
                />
              </div>
              <button
                onClick={handleUpload}
                className="btn btn-outline btn-primary mt-10"
              >
                Upload
              </button>
              <div className="mt-10">
                {imageUrls.map((url, index) => (
                  <div key={index} className="mb-4 text-center">
                    <img
                      src={url}
                      alt={`Preview ${index + 1}`}
                      className="max-w-xs mx-auto"
                    />
                  </div>
                ))}
              </div>
              {predict && (
                <div className="mt-12 p-5 bg-black border border-gray-300 rounded-lg">
                  <pre className="text-lg font-semibold text-gray-400 whitespace-pre-wrap fon">
                    <code>{predict}</code>
                  </pre>
                </div>
              )}
              {predict && (
                <div className="mt-10">
                  <button
                    onClick={handleReupload}
                    className="btn btn-outline btn-primary mt-10"
                  >
                    Re-upload with Prediction
                  </button>
                </div>
              )}
              {secondApiOutput && (
                <div className="mt-10 p-5 bg-blue-100 border border-blue-300 rounded-lg">
                  <pre className="text-lg font-semibold text-blue-700 whitespace-pre-wrap">
                    <code>{secondApiOutput}</code>
                  </pre>
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
