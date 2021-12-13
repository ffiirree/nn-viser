<template>
    <div class="page">
        <div class="menu">
            <div class="item">
                <div class="title">model</div>
                <el-select class='value' size="small" v-model="params.model" filterable @change="getLayers">
                    <el-option v-for="model in models" :key="model" :value='model'/>
                </el-select>
            </div>
            <div class="item">
                <div class="title">input</div>
                <el-select class="value" size="small" v-model="params.input" @change="params.target = images[params.input]">
                    <el-option v-for="image in Object.keys(images)" :key="images[image]" :value='image'/>
                </el-select>
            </div>
            <div class="item">
                <div class="title">layer</div>
                <el-select class="value" size="small" v-model="params.layer" @change="update">
                    <el-option v-for="(layer, index) in layers" :value="index" :key="index">{{index}} - {{layer.name}}/{{layer.layer}}</el-option>
                </el-select>
            </div>            <div class="item"><div class="title">activation</div><el-input class='value' size="small" v-model="params.activation"  @change="update"/></div>
            <div class="item"><div class="title">epochs</div><el-input class='value' size="small" v-model="params.epochs"  @change="update"/></div>
            <div class="item"><div class="title">lr</div><el-input class='value' size="small" v-model="params.lr"  @change="update"/></div>
            <div class="item"><el-checkbox class='button' v-model="params.clamp" @change="update">clamp</el-checkbox></div>
            <div class="item"><div class="title"></div><el-button icon='el-icon-refresh' type="primary" size="small" circle  @click="update"/></div>
        </div>
        <div class="network">
            <div class="iter">
                <img :src="res.output" width="256" height="256"/>
                <div>epoch = {{res.epoch}}, loss = {{res.loss}}</div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    data() {
        return {
            models: [],
            layers: {},
            images: {},
            res: {},
            params: {
                model: 'vgg19',
                input: '',
                layer: 34,
                activation: 94,
                epochs: 251,
                lr: 12,
                clamp: false,
            }
        };
    },
    created() {
        this.config()
    },
    sockets: {
        models(data) {
            this.models = data
        },
        layers(data) {
            this.layers = data
        },
        images(data) {
            this.images = data

            this.params.input = Object.keys(data)[0]
        },
        response_deep_dream(data) {
            this.res = data
        }
    },
    methods: {
        config() {
            this.$socket.emit('get_models')
            this.$socket.emit('get_images')
            this.getLayers()
        },
        getLayers() {
            this.$socket.emit('get_layers', { model: this.params.model })
        },
        update() {
            this.$socket.emit("deep_dream", this.params);
        },
    }
};
</script>

<style rel="stylesheet/scss" lang="scss" scoped>
.page {
    .network {
        .iter {
            display: flex;
            flex-flow: column;
            align-items: center;
        }
    }
}

</style>
