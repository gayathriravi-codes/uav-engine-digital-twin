import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'

function Piston({ position }) {
  const ref = useRef()
  useFrame((state) => {
    ref.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 3) * 0.3
  })
  return (
    <mesh ref={ref} position={position}>
      <cylinderGeometry args={[0.3, 0.3, 1, 16]} />
      <meshStandardMaterial color="silver" metalness={0.8} roughness={0.3} />
    </mesh>
  )
}

function Engine() {
  const fanRef = useRef()
  useFrame(() => {
    fanRef.current.rotation.z += 0.05
  })

  return (
    <group>
      {/* Engine block */}
      <mesh position={[0, 0, 0]}>
        <cylinderGeometry args={[1.2, 1.2, 2, 32]} />
        <meshStandardMaterial color="orange" metalness={0.5} roughness={0.4} />
      </mesh>

      {/* Pistons on top, animated */}
      <Piston position={[-0.6, 1.5, 0]} />
      <Piston position={[0, 1.5, 0]} />
      <Piston position={[0.6, 1.5, 0]} />

      {/* Spinning fan/turbine at front */}
      <mesh ref={fanRef} position={[0, 0, 1.3]}>
        <torusGeometry args={[0.8, 0.15, 16, 32]} />
        <meshStandardMaterial color="grey" metalness={0.9} roughness={0.2} />
      </mesh>
    </group>
  )
}

function App() {
  return (
    <Canvas style={{ height: '100vh', background: '#111' }} camera={{ position: [4, 3, 5] }}>
      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 5, 5]} />
      <Engine />
      <OrbitControls />
    </Canvas>
  )
}

export default App