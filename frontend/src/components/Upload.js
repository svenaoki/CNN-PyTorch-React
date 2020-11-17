import React, { useState } from "react";
import axios from "axios";
const Upload = () => {
  const [file, setFile] = useState("");

  const handleImageChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    let form_data = new FormData();
    form_data.append("file", file);

    let url = "http://localhost:5000";
    axios
      .post(url, form_data, {
        headers: {
          "content-type": "multipart/form-data",
        },
      })
      .then((res) => {
        console.log(res.data);
      })
      .catch((err) => console.log(err));
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          id="image"
          accept="image/png, image/jpeg"
          onChange={handleImageChange}
          required
        />
        <input type="submit" />
      </form>
    </div>
  );
};

export default Upload;
