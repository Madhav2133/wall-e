function startVideo() {
    var videoContainer = document.getElementById('video_container');
    videoContainer.style.display = 'block';

    // fetch('/video_feed')
    // .then(response => {
    //     if (response.ok) {
    //         console.log('Inititated');
    //     } else {
    //         console.error('Failed to initiate');
    //     }
    // })
    // .catch(error => {
    //     console.error('Error:', error);
    // });
}

function endVideo() {
    var videoContainer = document.getElementById('video_container');
    videoContainer.style.display = 'none';

    // // Code to close the webcam
    // fetch('/close_webcam')
    //     .then(response => {
    //         if (response.ok) {
    //             console.log('Webcam closed');
    //         } else {
    //             console.error('Failed to close webcam');
    //         }
    //     })
    //     .catch(error => {
    //         console.error('Error:', error);
    //     });
}
