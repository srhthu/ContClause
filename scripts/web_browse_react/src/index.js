console.log('in index js')

import "https://unpkg.com/react@18/umd/react.development.js"
import  "https://unpkg.com/react-dom@18/umd/react-dom.development.js"

function MyComponent() {
    return <h1>Hello, React!</h1>;
}
const container = document.getElementById('root');
const root = ReactDOM.createRoot(container);
root.render(
<React.StrictMode>
    <MyComponent />
</React.StrictMode>

);

console.log(xx)
console.log(x)
console.log(App)