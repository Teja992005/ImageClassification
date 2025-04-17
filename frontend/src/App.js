import React, { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState("");

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return alert("Please upload an image");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      setPrediction(`Predicted Class: ${data.prediction}`);
    } catch (error) {
      console.error("Error uploading file:", error);
      setPrediction("Error making prediction");
    }
  };

  return (
    <div>
      <h1>Image Classification</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload</button>
      <p>{prediction}</p>
    </div>
  );
}

export default App;
