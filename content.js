chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "getText") {
    // Get all the text content from the body of the page
    let pageText = document.body.innerText;
    sendResponse({pageText: pageText});
  }
});
