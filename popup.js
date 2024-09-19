document.addEventListener('DOMContentLoaded', () => {
  document.getElementById("parseButton").addEventListener("click", () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.scripting.executeScript({
        target: { tabId: tabs[0].id },
        files: ['content.js']
      }, () => {
        chrome.tabs.sendMessage(tabs[0].id, { action: "getText" }, (response) => {
          if (response && response.pageText) {
            console.log(response.pageText);

            // Send the text to the Flask server
            fetch('http://127.0.0.1:5000/receive_text', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({ pageText: response.pageText })
            })
            .then(res => res.json())
            .then(data => {
              console.log(data.summarized_topic);

              const resultDisplay = document.getElementById('resultDisplay');
              if (resultDisplay) {
                // Set the commentary directly without "Topic"
                resultDisplay.textContent = data.summarized_topic;
                
                // Optionally set the background color based on classification
                document.body.style.backgroundColor = data.background; // Use the background color returned from backend
              } else {
                console.error("Element with id 'resultDisplay' not found.");
              }
            })
            .catch(error => {
              console.error('Error sending text to Flask:', error);
            });
          } else {
            console.error("Failed to retrieve page text.");
          }
        });
      });
    });
  });
});
