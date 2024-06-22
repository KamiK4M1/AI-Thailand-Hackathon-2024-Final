// components/ArrowIndicator.tsx
"use client"
import { useEffect, useState } from 'react';
import axios from 'axios';
import styles from './css/ArrowIndicator.module.css';

interface ApiResponse {
  value: number;
}

const ArrowIndicator = () => {
  const [apiValue, setApiValue] = useState<number | null>(null);

  useEffect(() => {
    // Function to fetch the API value using Axios
    const fetchApiValue = async () => {
      try {
        const response = await axios.get<ApiResponse>('http://127.0.0.1:8000/');
        setApiValue(response.data.value); // Access the value property from the response data
        console.log(response.data)
      } catch (error) {
        console.error('Error fetching API value:', error);
      }
    };

    // Fetch the API value on component mount
    fetchApiValue();

    // Optionally, set up a timer to poll the API periodically
    const interval = setInterval(fetchApiValue, 5000); // Fetch every 5 seconds

    // Cleanup the interval on component unmount
    return () => clearInterval(interval);
  }, []);

  return (
    <div className={styles.container}>
      <div className={`${styles.arrow} ${apiValue === 120 ? styles.glow : ''}`}>←</div>
      <div className={`${styles.circle} ${apiValue === 150 ? styles.glow : ''}`}>○</div>
      <div className={`${styles.arrow} ${apiValue === 110 ? styles.glow : ''}`}>→</div>
    </div>
  );
};

export default ArrowIndicator;